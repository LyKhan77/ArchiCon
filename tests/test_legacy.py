"""Small, read-only fixtures for historical DA3 result import."""
import hashlib
import json
from pathlib import Path
import tempfile
import os
import subprocess
import shutil
import sys
import unittest
from unittest.mock import patch

import numpy as np


class LegacyImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stream = self.root / "old" / "outputs" / "stream"
        self.cache = self.stream / "results_output"
        self.keys = self.root / "old" / "data" / "keyframes"
        self.viewer = self.root / "viewer"
        self.output = self.root / "output"
        for directory in (self.cache, self.keys, self.viewer):
            directory.mkdir(parents=True)
        self.source = self.root / "source.mov"
        self.source.write_bytes(b"synthetic source, image check intentionally disabled")
        (self.viewer / "scene.json").write_text(json.dumps({"units": "m", "scale_uncertainty": 0.1397}))
        self.count = 12
        for index in range(self.count):
            self.save_frame(index)
            (self.keys / f"keyframe_{index:05d}_{index * 0.5:.2f}s.jpg").write_bytes(b"fixture image")
        self.poses = np.repeat(np.eye(4)[None], self.count, axis=0)
        self.save_poses()

    def save_frame(self, index, **extra):
        arrays = dict(image=np.zeros((4, 3, 3), dtype=np.uint8),
                      depth=np.ones((4, 3), dtype=np.float32),
                      conf=np.full((4, 3), 0.5, dtype=np.float32),
                      intrinsics=np.array([[2., 0., 1.], [0., 2., 2.], [0., 0., 1.]]))
        arrays.update(extra)
        np.savez(self.cache / f"frame_{index}.npz", **arrays)

    def save_poses(self):
        np.savetxt(self.stream / "camera_poses.txt", self.poses.reshape(self.count, 16))

    def run_import(self):
        from room_reconstruction.legacy import import_legacy
        return import_legacy(self.stream, self.keys, self.viewer, self.source, self.output, spotcheck=False)

    def test_numeric_cache_order_and_approximate_filename_timestamps(self):
        manifest = self.run_import()
        self.assertEqual([f["frame_id"] for f in manifest["frames"]], list(range(12)))
        self.assertEqual(manifest["frames"][10]["timestamp_seconds"], 5.)
        self.assertEqual(manifest["mapping"]["status"], "restored_order_approximate_time")
        self.assertFalse(manifest["mapping"]["verified_pts"])
        self.assertEqual(manifest["mapping"]["timestamp_precision_seconds"], 0.01)
        self.assertEqual(manifest["raw_depth_units"], "relative")
        self.assertIsNone(manifest["metric_scale_factor"])
        self.assertEqual(manifest["viewer"]["metadata"]["units"], "m")
        self.assertIn("estimator uncertainty", (self.output / "baseline_review.md").read_text())
        self.assertTrue((self.output / "legacy_manifest.json").is_file())

    def test_stream_settings_are_read_from_hashed_actual_config(self):
        config = self.stream / "streaming_config.yaml"
        config.write_text("Model:\n  chunk_size: 17\n  overlap: 3\n  loop_enable: true\n")
        manifest = self.run_import()
        settings = manifest["known_stream_settings"]
        self.assertEqual(settings["chunk_size"], 17)
        self.assertEqual(settings["overlap"], 3)
        self.assertIs(settings["loop_enable"], True)
        self.assertEqual(settings["config"]["sha256"], hashlib.sha256(config.read_bytes()).hexdigest())
        self.assertEqual(settings["config_snapshot"]["Model"]["chunk_size"], 17)

    def test_stream_settings_are_unknown_when_config_absent(self):
        settings = self.run_import()["known_stream_settings"]
        for name in ("chunk_size", "overlap", "loop_enable"):
            self.assertIsNone(settings[name])
        self.assertEqual(settings["status"], "unknown_config_absent")

    def test_opencv_thread_count_is_bounded_for_module_users(self):
        import cv2
        with patch("room_reconstruction.legacy._review_images", return_value={}) as review:
            self.run_import()
        self.assertTrue(review.called)
        self.assertLessEqual(cv2.getNumThreads(), 4)

    def test_inputs_are_unchanged_and_hashes_recorded(self):
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob("*") if p.is_file()}
        manifest = self.run_import()
        self.assertEqual(manifest["frames"][0]["cache"]["sha256"], before[self.cache / "frame_0.npz"])
        for path, digest in before.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_rejects_missing_cache_and_count_mismatch(self):
        (self.cache / "frame_4.npz").unlink()
        with self.assertRaisesRegex(ValueError, "contiguous|count"):
            self.run_import()
        self.assertFalse(self.output.exists())

    def test_rejects_duplicate_cache_id(self):
        (self.cache / "frame_00.npz").write_bytes((self.cache / "frame_0.npz").read_bytes())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.run_import()

    def test_rejects_invalid_or_nonmonotonic_keyframe_times(self):
        original = self.keys / "keyframe_00005_2.50s.jpg"
        for name in ("keyframe_00005_nans.jpg", "keyframe_00005_-2.50s.jpg", "keyframe_00005_0.50s.jpg"):
            replacement = self.keys / name
            original.rename(replacement)
            with self.assertRaisesRegex(ValueError, "timestamp|filename"):
                self.run_import()
            replacement.rename(original)

    def test_rejects_pose_count_nonfinite_homogeneous_and_rotation(self):
        for variant in ("count", "nonfinite", "homogeneous", "rotation"):
            with self.subTest(variant=variant):
                poses = self.poses.copy()
                if variant == "count":
                    poses = poses[:-1]
                elif variant == "nonfinite":
                    poses[0, 0, 3] = np.nan
                elif variant == "homogeneous":
                    poses[0, 3, 0] = 1
                else:
                    poses[0, 0, 0] = 2
                np.savetxt(self.stream / "camera_poses.txt", poses.reshape(-1, 16))
                with self.assertRaisesRegex(ValueError, "pose"):
                    self.run_import()

    def test_unknown_object_array_cannot_be_deserialized(self):
        self.save_frame(0, arbitrary=np.array([{"unsafe": True}], dtype=object))
        with self.assertRaisesRegex(ValueError, "Object arrays|object|pickle"):
            self.run_import()

    def test_frame_summary_and_single_open_archive(self):
        self.save_frame(0, depth=np.array([[np.nan, -1., 2.]] * 4, dtype=np.float32))
        real_load = np.load
        active = 0

        class Tracked:
            def __init__(self, archive):
                self.archive = archive
            def __enter__(self):
                nonlocal active
                active += 1
                self_test.assertEqual(active, 1)
                return self.archive
            def __exit__(self, *args):
                nonlocal active
                self.archive.close()
                active -= 1

        self_test = self
        def tracked_load(path, **kwargs):
            self.assertIs(kwargs.get("allow_pickle"), False)
            return Tracked(real_load(path, **kwargs))

        with patch("room_reconstruction.legacy.np.load", side_effect=tracked_load):
            manifest = self.run_import()
        summary = manifest["frames"][0]["arrays"]
        self.assertEqual(summary["image"]["shape"], [4, 3, 3])
        self.assertAlmostEqual(summary["depth"]["finite_fraction"], 2 / 3)
        self.assertAlmostEqual(summary["depth"]["positive_fraction"], 1 / 3)
        self.assertEqual(active, 0)

    def test_rejects_output_inside_any_source(self):
        self.output = self.stream / "new_manifest"
        with self.assertRaisesRegex(ValueError, "output|read.only"):
            self.run_import()

    def cli_fixture(self):
        (self.root / "configs").mkdir()
        config = self.root / "configs/paths.server.yaml"
        config.write_text("paths:\n" + "\n".join(
            f"  {key}: {value}" for key, value in {
                "project_root": self.root, "input_video": self.source,
                "legacy_stream": self.stream, "legacy_viewer": self.viewer,
                "legacy_project": self.root / "old"}.items()) + "\n")
        checkout = Path(__file__).resolve().parents[1]
        (self.root / "scripts").mkdir()
        (self.root / "src").symlink_to(checkout / "src", target_is_directory=True)
        script = self.root / "scripts/import_legacy.py"
        shutil.copyfile(checkout / "scripts/import_legacy.py", script)
        args = [sys.executable, str(script), "--run-id", "legacy-test",
                "--keyframes", str(self.keys), "--skip-image-check"]
        return args, config

    def test_cli_writes_run_provenance_and_refuses_existing_run(self):
        args, config = self.cli_fixture()
        first = subprocess.run(args, cwd=self.root.parent, capture_output=True, text=True, env=os.environ.copy())
        self.assertEqual(first.returncode, 0, first.stderr)
        run = self.root / "outputs/legacy-test"
        self.assertTrue((run / "legacy_manifest.json").exists())
        self.assertTrue((run / "run_provenance.json").exists())
        self.assertEqual(json.loads((run / "status.json").read_text())["status"], "complete")
        self.assertEqual(json.loads((run / "run_provenance.json").read_text())["config_path"], str(config.resolve()))
        previous = {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()}
        second = subprocess.run(args, cwd=self.root.parent, capture_output=True, text=True, env=os.environ.copy())
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual({p.name: p.read_bytes() for p in run.iterdir() if p.is_file()}, previous)

    def test_cli_corrupt_cache_persists_failed_status(self):
        args, _ = self.cli_fixture()
        (self.cache / "frame_0.npz").write_bytes(b"corrupted cache archive")
        result = subprocess.run(args, cwd=self.root.parent, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        status = json.loads((self.root / "outputs/legacy-test/status.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertTrue(status["reason"])

    def test_cli_late_provenance_error_marks_existing_manifest_failed(self):
        args, config = self.cli_fixture()
        with config.open("a") as stream:
            stream.write("unserializable_date: 2026-10-14\n")
        result = subprocess.run(args, cwd=self.root.parent, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        run = self.root / "outputs/legacy-test"
        self.assertTrue((run / "legacy_manifest.json").exists())
        status = json.loads((run / "status.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertIn("serializable", status["reason"])

    def test_image_spotcheck_samples_nine_without_claiming_exact_proof(self):
        import cv2
        from room_reconstruction.legacy import import_legacy

        self.source = self.root / "fixture.avi"
        writer = cv2.VideoWriter(str(self.source), cv2.VideoWriter_fourcc(*"MJPG"), 4., (64, 48))
        self.assertTrue(writer.isOpened())
        rng = np.random.default_rng(17)
        try:
            for index in range(24):
                frame = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
                writer.write(frame)
                if index % 2 == 0:
                    cv2.imwrite(str(self.keys / f"keyframe_{index // 2:05d}_{index / 4:.2f}s.jpg"), frame)
        finally:
            writer.release()
        manifest = import_legacy(self.stream, self.keys, self.viewer, self.source, self.output)
        check = manifest["image_spotcheck"]
        self.assertEqual(check["sample_count"], 9)
        self.assertFalse(check["exact_source_frame_proof"])
        self.assertEqual(check["status"], "sampled_inference_only")
        self.assertTrue(all(sample["grayscale_crop_correlation"] > .9 for sample in check["samples"]))
        self.assertTrue((self.output / "legacy_contact_sheet.jpg").exists())

    def test_cli_rejects_source_root_before_creating_output(self):
        config = self.root / "unsafe.yaml"
        config.write_text("paths:\n" + "\n".join(
            f"  {key}: {value}" for key, value in {
                "project_root": self.stream, "input_video": self.source,
                "legacy_stream": self.stream, "legacy_viewer": self.viewer,
                "legacy_project": self.root / "old"}.items()) + "\n")
        script = Path(__file__).resolve().parents[1] / "scripts/import_legacy.py"
        result = subprocess.run([sys.executable, str(script), "--config", str(config), "--run-id", "unsafe",
                                 "--keyframes", str(self.keys), "--skip-image-check"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("project_root", result.stderr)
        self.assertFalse((self.stream / "outputs").exists())


if __name__ == "__main__":
    unittest.main()
