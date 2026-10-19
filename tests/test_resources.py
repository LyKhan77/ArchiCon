"""Resource policy tests need neither CUDA nor third-party packages."""
import importlib.util
import io
import os
import json
from pathlib import Path
import sys
import unittest
from unittest import mock
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SPEC = importlib.util.find_spec("room_reconstruction.resources")
if SPEC:
    from room_reconstruction import resources as r
else:
    r = None

GIB = 1024 ** 3
UUID0 = "GPU-11111111-1111-1111-1111-111111111111"
UUID1 = "GPU-22222222-2222-2222-2222-222222222222"
CSV = f"0, {UUID0}, NVIDIA RTX 5090, 32768, 24576, 0\n1, {UUID1}, NVIDIA RTX 5090, 32768, 20480, 0\n"


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(r, "resource policy module must exist")
        self.gpus = r.parse_gpu_csv(CSV)
        self.policy = r.ResourcePolicy.from_mapping({
            "min_gpu_free_gib": 14, "min_host_available_gib": 8,
            "min_disk_free_gib": 50, "cpu_threads": 4, "workers": 2,
        })

    def evaluate(self, **kwargs):
        arguments = dict(gpus=self.gpus, policy=self.policy, selector=UUID1,
                         mem_available_bytes=9 * GIB, disk_free_bytes=60 * GIB,
                         busy_uuids=set(), current_visible=None)
        arguments.update(kwargs)
        return r.evaluate_resources(**arguments)

    def test_parse_csv_converts_mib_to_bytes(self):
        self.assertEqual(self.gpus[1].free_bytes, 20 * GIB)
        self.assertEqual(self.gpus[0].index, 0)

    def test_numeric_selector_is_physical_even_when_visibility_reorders(self):
        self.assertEqual(r.select_gpu(self.gpus, "1", "1,0").uuid, UUID1)
        with self.assertRaises(ValueError):
            r.select_gpu(self.gpus, "0", "1")

    def test_reject_hidden_requested_device(self):
        for visibility in ("0", UUID0, "", "-1", "GPU-unknown"):
            with self.subTest(visibility=visibility), self.assertRaises(ValueError):
                r.select_gpu(self.gpus, UUID1, visibility)

    def test_multiple_visible_devices_are_isolated_to_one_uuid(self):
        result = self.evaluate(current_visible="0,1")
        self.assertTrue(result["resource_ready"])
        self.assertEqual(result["launch_environment"]["CUDA_VISIBLE_DEVICES"], UUID1)
        self.assertEqual(result["logical_device"], "cuda:0")

    def test_environment_is_copied_and_thread_limited(self):
        original = {"CUDA_VISIBLE_DEVICES": "0,1", "KEEP": "value"}
        before = dict(os.environ)
        environment = r.launch_environment(self.gpus[1], self.policy, original)
        self.assertEqual(original["CUDA_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(dict(os.environ), before)
        self.assertEqual(environment["KEEP"], "value")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.assertEqual(environment[name], "4")

    def test_reject_insufficient_available_host_memory(self):
        self.assertFalse(self.evaluate(mem_available_bytes=8 * GIB - 1)["resource_ready"])

    def test_reject_insufficient_free_gpu_memory(self):
        gpus = r.parse_gpu_csv(CSV.replace("20480", "14335"))
        self.assertFalse(self.evaluate(gpus=gpus)["resource_ready"])

    def test_reject_insufficient_disk(self):
        self.assertFalse(self.evaluate(disk_free_bytes=50 * GIB - 1)["resource_ready"])

    def test_reject_busy_compute_process_with_zero_utilization(self):
        self.assertFalse(self.evaluate(busy_uuids={UUID1})["resource_ready"])
        self.assertTrue(self.evaluate(busy_uuids={UUID0})["resource_ready"])

    def test_reject_nonzero_utilization(self):
        gpus = r.parse_gpu_csv(CSV.replace("20480, 0", "20480, 1"))
        self.assertFalse(self.evaluate(gpus=gpus)["resource_ready"])

    def test_unknown_gpu_or_process_probe_fails_closed(self):
        self.assertFalse(self.evaluate(gpus=[], gpu_error="nvidia-smi missing")["resource_ready"])
        self.assertFalse(self.evaluate(busy_uuids=None)["resource_ready"])
        self.assertFalse(self.evaluate(mem_available_bytes=None)["resource_ready"])

    def test_malformed_csv_fails_closed(self):
        for value in ("", "No devices were found", CSV.replace("20480", "N/A"),
                      CSV.replace("20480", "-1"), CSV.replace("20480", "99999"),
                      CSV.replace(", 0\n", ", nan\n"), CSV + CSV,
                      CSV.replace(UUID1, "GPU-bad")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                r.parse_gpu_csv(value)

    def test_select_requires_exactly_one_known_gpu(self):
        for selector in ("0,1", "", "2", "GPU-2222"):
            with self.subTest(selector=selector), self.assertRaises(ValueError):
                r.select_gpu(self.gpus, selector)

    def test_mem_available_uses_kib_not_total_or_free(self):
        self.assertEqual(r.parse_mem_available("MemTotal: 900000 kB\nMemFree: 10 kB\nMemAvailable: 12345 kB\n"), 12345 * 1024)
        for value in ("MemTotal: 900000 kB\n", "MemAvailable: 8 MB", "MemAvailable: -1 kB"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                r.parse_mem_available(value)

    def test_process_uuid_output_is_strict_and_empty_is_idle(self):
        self.assertEqual(r.parse_compute_process_uuids(""), set())
        self.assertEqual(r.parse_compute_process_uuids(f"{UUID1}\n{UUID1}\n"), {UUID1})
        with self.assertRaises(ValueError):
            r.parse_compute_process_uuids("N/A")

    def test_invalid_policy_rejected(self):
        for mapping in ({"cpu_threads": 0}, {"workers": -1}, {"min_gpu_free_gib": "nan"}):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                r.ResourcePolicy.from_mapping(mapping)

    def test_local_profile_can_pass_without_gpu_probe(self):
        result = self.evaluate(gpus=[], selector=None, gpu_required=False)
        self.assertTrue(result["resource_ready"])
        self.assertIsNone(result["logical_device"])


class PreflightTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "scripts" / "preflight.py"
        self.assertTrue(path.exists(), "preflight CLI must exist")
        spec = importlib.util.spec_from_file_location("preflight_cli", path)
        self.cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.cli)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        (self.root / "configs").mkdir()
        for stream in ("stdout", "stderr"):
            patcher = mock.patch.object(sys, stream, io.StringIO())
            patcher.start()
            self.addCleanup(patcher.stop)

    def registry(self, expected=4):
        path = self.root / "configs" / "models.yaml"
        path.write_text("schema_version: 1\nmodels:\n  pilot:\n    model_id: test/model\n"
                        "    cache_directory: models--test--model\n"
                        "    revision: " + "a" * 40 + "\n    weights_bytes: " + str(expected) + "\n")
        snapshot = self.root / "cache" / "models--test--model" / "snapshots" / ("a" * 40)
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text('{"model_type": "fixture"}')
        (snapshot / "model.safetensors").write_bytes(b"data")
        return path, snapshot

    def test_registry_reports_file_validation_without_load_claim(self):
        path, snapshot = self.registry()
        result = self.cli.checkpoint_registry(path, self.root / "cache")
        self.assertTrue(result["models"]["pilot"]["files_ready"])
        self.assertEqual(result["verification_scope"], "file_presence_config_and_size_only")
        (snapshot / "model.safetensors").write_bytes(b"bad")
        result = self.cli.checkpoint_registry(path, self.root / "cache")
        self.assertFalse(result["models"]["pilot"]["files_ready"])
        (snapshot / "config.json").write_text("invalid JSON")
        result = self.cli.checkpoint_registry(path, self.root / "cache")
        self.assertFalse(result["models"]["pilot"]["config_valid"])

    def test_registry_rejects_path_traversal(self):
        path, _ = self.registry()
        path.write_text(path.read_text().replace("models--test--model", "../outside"))
        result = self.cli.checkpoint_registry(path, self.root / "cache")
        self.assertFalse(result["models"]["pilot"]["files_ready"])

    def test_cli_writes_actual_root_and_separates_readiness(self):
        self.registry()
        config = self.root / "configs" / "paths.server.yaml"
        config.write_text(f"profile: server\npaths:\n  project_root: /untrusted/reference\n"
                          f"  outputs: /untrusted/outputs\n  model_cache: cache\n"
                          f"compute:\n  gpu_selector: {UUID1}\n")

        def probe(argv):
            stdout = CSV if any("query-gpu=" in item for item in argv) else ""
            return {"argv": argv, "returncode": 0, "stdout": stdout, "stderr": "", "error": None}

        with mock.patch.object(self.cli, "ROOT", self.root), \
             mock.patch.object(self.cli, "probe_command", side_effect=probe), \
             mock.patch.object(self.cli, "host_memory", return_value={"available_bytes": 9 * GIB, "error": None}), \
             mock.patch.object(self.cli.shutil, "disk_usage", return_value=mock.Mock(free=60 * GIB)), \
             mock.patch.object(self.cli, "package_inventory", return_value={"PyYAML": "6.0.3"}), \
             mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.cli.main(["--config", str(config), "--run-id", "fixture"]), 0)
            report = json.loads((self.root / "outputs" / "fixture" / "preflight.json").read_text())
            self.assertEqual(report["source_root"], str(self.root))
            self.assertTrue(report["resource_ready"])
            self.assertFalse(report["inference_ready"])
            self.assertIn("torch", report["inference"]["missing_packages"])
            self.assertEqual(report["cpu_thread_policy"]["threads"], 4)
            self.assertFalse(report["legacy_sources"]["legacy_project"]["available"])
            self.assertTrue(report["limitations"])
            status_path = self.root / "outputs" / "fixture" / "status.json"
            self.assertTrue(status_path.exists(), "reserved successful run must have status.json")
            status_before = status_path.read_bytes()
            self.assertEqual(json.loads(status_before)["status"], "resource_passed")
            self.assertEqual(self.cli.main(["--config", str(config), "--run-id", "fixture"]), 2)
            self.assertEqual(status_path.read_bytes(), status_before)

    def test_missing_nvidia_query_produces_report_and_exit_two(self):
        self.registry()
        config = self.root / "configs" / "paths.server.yaml"
        config.write_text(f"profile: server\npaths:\n  model_cache: cache\ncompute:\n  gpu_selector: {UUID1}\n")
        failure = {"returncode": None, "stdout": "", "stderr": "", "error": "not found"}
        with mock.patch.object(self.cli, "ROOT", self.root), \
             mock.patch.object(self.cli, "probe_command", return_value=failure), \
             mock.patch.object(self.cli, "host_memory", return_value={"available_bytes": 9 * GIB, "error": None}), \
             mock.patch.object(self.cli.shutil, "disk_usage", return_value=mock.Mock(free=60 * GIB)):
            self.assertEqual(self.cli.main(["--config", str(config), "--run-id", "failed"]), 2)
        report = json.loads((self.root / "outputs" / "failed" / "preflight.json").read_text())
        self.assertFalse(report["resource_ready"])
        self.assertFalse(report["inference_ready"])
        status_path = self.root / "outputs" / "failed" / "status.json"
        self.assertTrue(status_path.exists(), "resource rejection must have status.json")
        status = json.loads(status_path.read_text())
        self.assertEqual(status["status"], "resource_rejected")
        self.assertTrue(status["reasons"])

    def test_unexpected_post_reservation_failure_is_persisted(self):
        config = self.root / "configs" / "paths.local.yaml"
        config.write_text("profile: local\npaths: {}\ncompute: {}\n")
        with mock.patch.object(self.cli, "ROOT", self.root), \
             mock.patch.object(self.cli, "package_inventory", side_effect=RuntimeError("inventory fixture failure")):
            try:
                result = self.cli.main(["--config", str(config), "--run-id", "broken"])
            except Exception as exc:
                result = f"unexpected uncaught {type(exc).__name__}"
            self.assertEqual(result, 2)
        run_dir = self.root / "outputs" / "broken"
        status = json.loads((run_dir / "status.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["failure_reason"], "inventory fixture failure")
        self.assertEqual(status["error_type"], "RuntimeError")
        self.assertFalse((run_dir / "preflight.json").exists())

    def test_legacy_sources_snapshot_both_roots_without_model_import(self):
        self.assertTrue(hasattr(self.cli, "legacy_source_snapshots"), "legacy snapshot collector must exist")
        legacy = self.root / "legacy"
        da3 = legacy / "Depth-Anything-3"
        da3.mkdir(parents=True)

        def snapshot(path):
            return {"root": str(path), "available": True, "dirty": True,
                    "source_files": {"da3_streaming/loop_utils/loop_detector.py": {"sha256": "fixture", "size_bytes": 12}}}

        with mock.patch.object(self.cli, "git_snapshot", side_effect=snapshot) as collect:
            sources, limitations = self.cli.legacy_source_snapshots(self.root, {"legacy_project": str(legacy)})
        self.assertEqual(collect.call_args_list, [mock.call(legacy), mock.call(da3)])
        self.assertTrue(sources["legacy_project"]["available"])
        self.assertTrue(sources["depth_anything_3"]["dirty"])
        self.assertIn("da3_streaming/loop_utils/loop_detector.py", sources["depth_anything_3"]["source_files"])
        self.assertEqual(limitations, [])

    def test_missing_legacy_sources_are_limitations(self):
        self.assertTrue(hasattr(self.cli, "legacy_source_snapshots"), "legacy snapshot collector must exist")
        sources, limitations = self.cli.legacy_source_snapshots(self.root, {"legacy_project": "missing"})
        self.assertFalse(sources["legacy_project"]["available"])
        self.assertFalse(sources["depth_anything_3"]["available"])
        self.assertEqual(len(limitations), 2)

    def test_legacy_snapshot_probe_errors_are_limitations(self):
        legacy = self.root / "legacy"
        (legacy / "Depth-Anything-3").mkdir(parents=True)
        with mock.patch.object(self.cli, "git_snapshot", side_effect=RuntimeError("git fixture failure")):
            sources, limitations = self.cli.legacy_source_snapshots(self.root, {"legacy_project": str(legacy)})
        self.assertEqual(len(limitations), 2)
        self.assertEqual(sources["depth_anything_3"]["error"], "git fixture failure")

    def test_cuda_12_compiler_warning_does_not_change_resource_gate(self):
        self.registry()
        config = self.root / "configs" / "paths.server.yaml"
        config.write_text(f"profile: server\npaths:\n  model_cache: cache\ncompute:\n  gpu_selector: {UUID1}\n")

        def probe(argv):
            stdout = CSV if any("query-gpu=" in item for item in argv) else ""
            if "--version" in argv:
                stdout = "Cuda compilation tools, release 12.0, V12.0.140"
            return {"argv": argv, "returncode": 0, "stdout": stdout, "stderr": "", "error": None}

        with mock.patch.object(self.cli, "ROOT", self.root), \
             mock.patch.object(self.cli, "probe_command", side_effect=probe), \
             mock.patch.object(self.cli.shutil, "which", return_value="/fixture/nvcc"), \
             mock.patch.object(self.cli, "host_memory", return_value={"available_bytes": 9 * GIB, "error": None}), \
             mock.patch.object(self.cli.shutil, "disk_usage", return_value=mock.Mock(free=60 * GIB)), \
             mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.cli.main(["--config", str(config), "--run-id", "compiler"]), 0)
        report = json.loads((self.root / "outputs" / "compiler" / "preflight.json").read_text())
        self.assertTrue(any("CUDA 12.0" in warning for warning in report["warnings"]))
        self.assertFalse(report["inference_ready"])


if __name__ == "__main__":
    unittest.main()
