#!/usr/bin/env python3
"""Record resource availability without loading models or allocating GPU tensors."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from room_reconstruction.provenance import (create_run_dir, git_snapshot, load_yaml,
                                          sha256_file, write_json)
from room_reconstruction.resources import (ResourcePolicy, evaluate_resources,
    parse_compute_process_uuids, parse_gpu_csv, parse_mem_available)


def probe_command(argv):
    try:
        process = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        return {"argv": argv, "returncode": process.returncode, "stdout": process.stdout,
                "stderr": process.stderr, "error": None}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "returncode": None, "stdout": "", "stderr": "", "error": str(exc)}


def host_memory():
    try:
        return {"available_bytes": parse_mem_available(Path("/proc/meminfo").read_text()),
                "source": "/proc/meminfo:MemAvailable", "error": None}
    except (OSError, ValueError) as exc:
        return {"available_bytes": None, "source": "/proc/meminfo:MemAvailable", "error": str(exc)}


def package_inventory():
    return dict(sorted((dist.metadata["Name"], dist.version)
                       for dist in importlib.metadata.distributions() if dist.metadata["Name"]))


def legacy_source_snapshots(root, paths):
    sources, limitations = {}, []
    configured = paths.get("legacy_project")
    legacy = Path(configured) if isinstance(configured, str) and configured else None
    if legacy is not None and not legacy.is_absolute():
        legacy = root / legacy
    candidates = {"legacy_project": legacy,
                  "depth_anything_3": legacy / "Depth-Anything-3" if legacy else None}
    for name, path in candidates.items():
        try:
            if path is None or not path.is_dir():
                raise ValueError("Source checkout is missing or unconfigured")
            snapshot = git_snapshot(path)
            if not snapshot.get("available"):
                raise ValueError("Git snapshot unavailable for source checkout")
            sources[name] = snapshot
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            sources[name] = {"root": str(path) if path else None, "available": False, "error": str(exc)}
            limitations.append(f"{name}: {exc}")
    return sources, limitations


def checkpoint_registry(registry_path, cache_path):
    result = {"verification_scope": "file_presence_config_and_size_only", "models": {}}
    try:
        registry = load_yaml(registry_path)
        if registry.get("schema_version") != 1 or not isinstance(registry.get("models"), dict) or not registry["models"]:
            raise ValueError("Model registry requires schema_version 1 and nonempty models mapping")
        result["registry_sha256"] = sha256_file(registry_path)
        for name, entry in registry["models"].items():
            state = {"files_ready": False, "config_valid": False, "errors": []}
            result["models"][name] = state
            try:
                if not isinstance(entry, dict):
                    raise ValueError("Model entry must be a mapping")
                directory, revision, size = (entry.get(key) for key in ("cache_directory", "revision", "weights_bytes"))
                if not isinstance(directory, str) or not re.fullmatch(r"models--[A-Za-z0-9._-]+", directory):
                    raise ValueError("Invalid cache_directory")
                if not isinstance(revision, str) or not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
                    raise ValueError("Revision must be a pinned 40-character commit")
                if isinstance(size, bool) or not isinstance(size, int) or size <= 0 or not isinstance(entry.get("model_id"), str):
                    raise ValueError("Model ID and positive integer weights_bytes required")
                snapshot = cache_path / directory / "snapshots" / revision
                config, weights = snapshot / "config.json", snapshot / "model.safetensors"
                state.update(model_id=entry["model_id"], revision=revision, snapshot=str(snapshot),
                             expected_weights_bytes=size, config_present=config.is_file(),
                             weights_present=weights.is_file())
                try:
                    configuration = json.loads(config.read_text())
                    if not isinstance(configuration, dict) or not configuration:
                        raise ValueError("config.json must contain a nonempty object")
                    state.update(config_valid=True, config_sha256=sha256_file(config))
                except (OSError, ValueError) as exc:
                    state["errors"].append(f"Config validation failed: {exc}")
                actual = weights.stat().st_size if weights.is_file() else None
                state["actual_weights_bytes"] = actual
                if actual != size:
                    state["errors"].append("Weights missing or byte size differs from registry")
                state["files_ready"] = not state["errors"]
            except (OSError, ValueError) as exc:
                state["errors"].append(str(exc))
    except (OSError, ValueError) as exc:
        result["error"] = str(exc)
    return result


def collect_report(root, config_path, config, policy, run_dir, run_id):
    compute, paths = config.get("compute", {}), config.get("paths", {})
    memory = host_memory()
    try:
        disk = {"path": str(run_dir), "free_bytes": shutil.disk_usage(run_dir).free, "error": None}
    except OSError as exc:
        disk = {"path": str(run_dir), "free_bytes": None, "error": str(exc)}
    gpu_probe = probe_command(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu", "--format=csv,noheader,nounits"])
    process_probe = probe_command(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"])
    gpus, busy, gpu_error = [], None, None
    try:
        if gpu_probe["returncode"] != 0:
            raise ValueError(gpu_probe["error"] or gpu_probe["stderr"] or "GPU query failed")
        gpus = parse_gpu_csv(gpu_probe["stdout"])
    except ValueError as exc:
        gpu_error = str(exc)
    if process_probe["returncode"] == 0:
        try:
            busy = parse_compute_process_uuids(process_probe["stdout"])
        except ValueError as exc:
            process_probe["error"] = str(exc)
    required = config.get("profile") != "local" or compute.get("gpu_enabled", False)
    resources = evaluate_resources(gpus=gpus, policy=policy, selector=compute.get("gpu_selector"),
        mem_available_bytes=memory["available_bytes"], disk_free_bytes=disk["free_bytes"],
        busy_uuids=busy, current_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu_error=gpu_error, gpu_required=required)
    tools = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe", "nvcc")}
    nvcc = probe_command([tools["nvcc"], "--version"]) if tools["nvcc"] else None
    packages = package_inventory()
    installed = {re.sub(r"[-_.]+", "-", name).lower() for name in packages}
    missing = sorted({"torch", "depth-anything-3"} - installed)
    cache_path = Path(paths.get("model_cache", "models/huggingface"))
    if not cache_path.is_absolute():
        cache_path = root / cache_path
    checkpoints = checkpoint_registry(root / "configs" / "models.yaml", cache_path)
    legacy_sources, limitations = legacy_source_snapshots(root, paths)
    warnings = ["CUDA runtime, GPU architecture support, and compiled kernels are not validated.",
                "Model file validation does not establish model loading or inference correctness."]
    for name in ("ffmpeg", "ffprobe", "nvcc"):
        if not tools[name]:
            warnings.append(f"{name} is not on PATH.")
    if nvcc and re.search(r"release\s+12\.0\b", nvcc["stdout"]) and any(re.search(r"Blackwell|RTX.*50\d{2}", gpu.name, re.I) for gpu in gpus):
        warnings.append("CUDA 12.0 compiler cannot build native Blackwell kernels; use a validated compatible toolchain.")
    report = {
        "schema_version": 1, "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_root": str(root),
        "configuration": {"path": str(config_path.resolve()), "sha256": sha256_file(config_path),
                          "declared_project_root": paths.get("project_root"),
                          "declared_outputs": paths.get("outputs"), "profile": config.get("profile")},
        "source": git_snapshot(root), "legacy_sources": legacy_sources,
        "platform": {"system": platform.system(),
            "release": platform.release(), "machine": platform.machine(), "cpu": platform.processor(),
            "cpu_count": os.cpu_count(), "python": sys.version, "executable": sys.executable},
        "memory": memory, "disk": disk, "gpu_probe": gpu_probe, "process_probe": process_probe,
        "gpus": [asdict(gpu) for gpu in gpus], "compute_process_uuids": sorted(busy) if busy is not None else None,
        "current_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "resources": resources, "resource_ready": resources["resource_ready"],
        "cpu_thread_policy": {"threads": policy.cpu_threads, "workers": policy.workers,
            "current_environment": {name: os.environ.get(name) for name in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}},
        "toolchain": {"paths": tools, "nvcc_version": nvcc}, "packages": packages,
        "checkpoints": checkpoints, "inference_ready": False,
        "inference": {"missing_packages": missing, "runtime_validated": False,
                      "model_load_validated": False, "reason": "Inference is unvalidated; preflight performs no model import or GPU allocation."},
        "warnings": warnings, "limitations": limitations,
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paths.server.yaml")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    root, config_path = ROOT.resolve(), Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    try:
        config = load_yaml(config_path)
        compute, paths = config.get("compute", {}), config.get("paths", {})
        if not isinstance(compute, dict) or not isinstance(paths, dict):
            raise ValueError("compute and paths must be mappings")
        policy = ResourcePolicy.from_mapping(compute)
        run_dir = create_run_dir(root, args.run_id)
    except (OSError, ValueError) as exc:
        print(f"preflight: {exc}", file=sys.stderr)
        return 2
    try:
        report = collect_report(root, config_path, config, policy, run_dir, args.run_id)
        write_json(run_dir / "preflight.json", report)
        ready = report["resource_ready"]
        write_json(run_dir / "status.json", {"status": "resource_passed" if ready else "resource_rejected",
            "resource_ready": ready, "inference_ready": False, "reasons": report["resources"]["errors"]})
        print(f"resource_ready={str(ready).lower()} inference_ready=false report={run_dir / 'preflight.json'}")
        for error in report["resources"]["errors"]:
            print(f"gate: {error}")
        return 0 if ready else 2
    except Exception as exc:
        try:
            write_json(run_dir / "status.json", {"status": "failed", "failure_reason": str(exc),
                "error_type": type(exc).__name__, "resource_ready": False, "inference_ready": False})
        except OSError as status_error:
            print(f"preflight: could not persist failure status: {status_error}", file=sys.stderr)
        print(f"preflight failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
