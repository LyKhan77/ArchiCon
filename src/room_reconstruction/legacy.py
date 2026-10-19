"""Read-only import of cached DA3-Streaming outputs, without scale conversion.

API: import_legacy(stream_dir, keyframes_dir, viewer_dir, source_video,
                   output_dir, spotcheck=True) -> JSON-compatible manifest.
Inputs remain references; only output_dir receives generated review artifacts.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

import numpy as np

from .provenance import load_yaml, sha256_file, write_json


def _reference(path: Path) -> dict:
    return {"path": str(path.absolute()), "resolved_path": str(path.resolve()),
            "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _ordered_frames(cache_dir: Path, keyframes_dir: Path) -> tuple[list, list]:
    cached = []
    for path in cache_dir.glob("frame_*.npz"):
        match = re.fullmatch(r"frame_(\d+)\.npz", path.name)
        if not match:
            raise ValueError(f"Invalid cache filename: {path.name}")
        cached.append((int(match[1]), path))
    cached.sort(key=lambda item: item[0])
    ids = [item[0] for item in cached]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate cached frame IDs")
    if not ids or ids != list(range(len(ids))):
        raise ValueError("Cached frame IDs must be contiguous from zero")
    keys = []
    # Match the original producer's sorted image paths, never timestamp sort.
    for path in sorted(keyframes_dir.glob("*.jpg")):
        match = re.fullmatch(r"keyframe_(\d{5})_(\d+\.\d{2})s\.jpg", path.name)
        if not match:
            raise ValueError(f"Invalid keyframe filename/timestamp: {path.name}")
        timestamp = float(match[2])
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError(f"Invalid timestamp: {path.name}")
        keys.append((int(match[1]), timestamp, path))
    key_ids = [item[0] for item in keys]
    if len(set(key_ids)) != len(key_ids):
        raise ValueError("duplicate keyframe IDs")
    if key_ids != list(range(len(keys))):
        raise ValueError("Keyframe IDs must be contiguous from zero in producer lexicographic order")
    if len(keys) != len(cached):
        raise ValueError(f"Frame count mismatch: {len(cached)} cached, {len(keys)} keyframes")
    if any(b[1] <= a[1] for a, b in zip(keys, keys[1:])):
        raise ValueError("Keyframe timestamps must increase strictly")
    return cached, keys


def _poses(path: Path, count: int) -> np.ndarray:
    values = np.loadtxt(path, ndmin=2)
    if values.shape != (count, 16):
        raise ValueError(f"Invalid pose count/shape {values.shape}; expected {(count, 16)}")
    poses = values.reshape(count, 4, 4)
    if not np.isfinite(poses).all():
        raise ValueError("Nonfinite camera pose")
    if not np.allclose(poses[:, 3, :], [0, 0, 0, 1], atol=1e-5, rtol=0):
        raise ValueError("Invalid homogeneous camera pose")
    rotations = poses[:, :3, :3]
    if not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-3, rtol=0):
        raise ValueError("Camera pose rotation is not orthonormal")
    if not np.allclose(np.linalg.det(rotations), 1, atol=1e-3, rtol=0):
        raise ValueError("Camera pose rotation is not proper")
    return poses


def _array_summary(array: np.ndarray) -> dict:
    if array.dtype.kind not in "buif":
        raise ValueError(f"Only real numeric arrays allowed; found {array.dtype}")
    finite = np.isfinite(array)
    values = array[finite]
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "finite_fraction": float(finite.mean()) if array.size else 0.,
            "positive_fraction": float(np.count_nonzero(finite & (array > 0)) / array.size) if array.size else 0.,
            "min": float(values.min()) if values.size else None,
            "max": float(values.max()) if values.size else None}


def _inspect_cache(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        required = {"image", "depth", "conf", "intrinsics"}
        if not required.issubset(archive.files):
            raise ValueError(f"Missing required arrays in {path}: {required - set(archive.files)}")
        summaries = {}
        # Materialize one array at a time, and close each archive before the next frame.
        for name in archive.files:
            array = archive[name]
            summaries[name] = _array_summary(array)
            if name == "image" and (array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8):
                raise ValueError(f"Invalid RGB uint8 image shape/type in {path}")
            if name == "intrinsics":
                if (array.shape != (3, 3) or not np.isfinite(array).all()
                        or array[0, 0] <= 0 or array[1, 1] <= 0
                        or not np.allclose(array[2], [0, 0, 1], atol=1e-5, rtol=0)):
                    raise ValueError(f"Invalid intrinsics in {path}")
                summaries[name]["matrix"] = array.tolist()
            del array
        image_shape = summaries["image"]["shape"][:2]
        if not all(image_shape) or any(summaries[name]["shape"] != image_shape for name in ("depth", "conf")):
            raise ValueError(f"Inconsistent image/depth/conf shape in {path}")
        return summaries


def _review_images(keys: list, source_video: Path, output_dir: Path, spotcheck: bool) -> dict:
    import cv2

    samples = sorted(set(np.linspace(0, len(keys) - 1, min(9, len(keys)), dtype=int).tolist()))
    report = {"enabled": spotcheck, "status": "skipped" if not spotcheck else "unavailable",
              "method": "Inference only: orientation-auto source frame round(filename_seconds * CAP_PROP_FPS); central grayscale crop correlation",
              "exact_source_frame_proof": False, "samples": []}
    capture = None
    tiles = []
    try:
        if spotcheck:
            capture = cv2.VideoCapture(str(source_video))
            if capture.isOpened():
                report["orientation_auto_set"] = bool(capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1))
                report["orientation_auto_readback"] = capture.get(cv2.CAP_PROP_ORIENTATION_AUTO)
                report["fps"] = float(capture.get(cv2.CAP_PROP_FPS))
                if math.isfinite(report["fps"]) and report["fps"] > 0:
                    report["status"] = "sampled_inference_only"
                else:
                    report["fps"] = None
        for index in samples:
            _, timestamp, path = keys[index]
            saved = cv2.imread(str(path))
            if saved is not None:
                tile = cv2.resize(saved, (168, 224))
                cv2.putText(tile, f"{index}: {timestamp:.2f}s", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
                tiles.append(tile)
            if report["status"] != "sampled_inference_only":
                continue
            source_index = round(timestamp * report["fps"])
            sought = capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
            ok, source = capture.read() if sought else (False, None)
            item = {"frame_id": index, "approximate_source_frame_index": source_index,
                    "timestamp_seconds": timestamp, "grayscale_crop_correlation": None}
            if ok and saved is not None:
                vectors = []
                for image in (saved, source):
                    height, width = image.shape[:2]
                    cropped = image[height // 10:height - height // 10, width // 10:width - width // 10]
                    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    vectors.append(cv2.resize(gray, (96, 96)).astype(np.float32).ravel())
                if all(float(vector.std()) > 0 for vector in vectors):
                    correlation = float(np.corrcoef(*vectors)[0, 1])
                    if math.isfinite(correlation):
                        item["grayscale_crop_correlation"] = correlation
            report["samples"].append(item)
        if tiles:
            while len(tiles) % 3:
                tiles.append(np.zeros_like(tiles[0]))
            sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
            target = output_dir / "legacy_contact_sheet.jpg"
            if not cv2.imwrite(str(target), sheet):
                raise OSError(f"Cannot write contact sheet: {target}")
            report["contact_sheet"] = target.name
    finally:
        if capture is not None:
            capture.release()
    report["sample_count"] = len(report["samples"])
    return report


def import_legacy(stream_dir: Path, keyframes_dir: Path, viewer_dir: Path,
                  source_video: Path, output_dir: Path, spotcheck: bool = True) -> dict:
    """Validate cached order/poses and write bounded summaries and a review report.

    Approximate legacy filename times are not decoder PTS. Raw depths and c2w
    translation remain relative, regardless of the processed viewer's units.
    """
    stream_dir, keyframes_dir, viewer_dir, source_video, output_dir = map(
        Path, (stream_dir, keyframes_dir, viewer_dir, source_video, output_dir))
    output_resolved = output_dir.resolve()
    for source in (stream_dir, keyframes_dir, viewer_dir):
        resolved = source.resolve()
        if output_resolved.is_relative_to(resolved) or resolved.is_relative_to(output_resolved):
            raise ValueError("Output must be separate from read-only legacy source directories")
    if output_resolved == source_video.resolve() or source_video.resolve().is_relative_to(output_resolved):
        raise ValueError("Output must be separate from source video")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    for source in (stream_dir, keyframes_dir, viewer_dir):
        if not source.is_dir():
            raise ValueError(f"Missing source directory: {source}")
    if not source_video.is_file():
        raise ValueError(f"Missing source video: {source_video}")
    import cv2
    cv2.setNumThreads(min(4, max(1, cv2.getNumThreads())))
    if cv2.getNumThreads() > 4:
        # Some backends ignore arbitrary limits; serial execution is portable.
        cv2.setNumThreads(0)
    cached, keys = _ordered_frames(stream_dir / "results_output", keyframes_dir)
    pose_path = stream_dir / "camera_poses.txt"
    poses = _poses(pose_path, len(cached))
    scene_path = viewer_dir / "scene.json"
    metadata = json.loads(scene_path.read_text()) if scene_path.is_file() else {}
    frames = []
    for (frame_id, cache), (_, timestamp, keyframe), pose in zip(cached, keys, poses):
        frames.append({"frame_id": frame_id, "timestamp_seconds": timestamp,
                       "cache": _reference(cache), "keyframe": _reference(keyframe),
                       "arrays": _inspect_cache(cache), "camera_to_world": pose.tolist()})
    old_root = stream_dir.resolve().parent.parent
    weights = old_root / "Depth-Anything-3/da3_streaming/weights/model.safetensors"
    model = {"family": "DA3-Streaming", "weights_reference": str(weights),
             "weights_resolved_path": str(weights.resolve()) if weights.exists() else None,
             "weights_available": weights.is_file(),
             "architecture_or_revision_verified": False}
    viewer_assets = [_reference(path) for path in sorted(viewer_dir.rglob("*"))
                     if path.is_file() and path.suffix.lower() in {".glb", ".ply", ".json", ".html"}]
    configs = [_reference(path) for pattern in ("*.yaml", "*.yml") for path in sorted(stream_dir.glob(pattern))]
    stream_settings = {"chunk_size": None, "overlap": None, "loop_enable": None,
                       "status": "unknown_config_absent", "config": None, "config_snapshot": None}
    actual_config = stream_dir / "streaming_config.yaml"
    if actual_config.is_file():
        snapshot = load_yaml(actual_config)
        model_settings = snapshot.get("Model", {})
        if not isinstance(model_settings, dict):
            raise ValueError("streaming_config.yaml Model must be a mapping")
        stream_settings.update({key: model_settings.get(key) for key in ("chunk_size", "overlap", "loop_enable")})
        stream_settings.update({"status": "read_from_actual_config", "config": _reference(actual_config),
                                "config_snapshot": snapshot})
    manifest = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_status": "visual_legacy_baseline_only", "frame_count": len(frames),
        "source_video": _reference(source_video), "camera_poses": _reference(pose_path),
        "mapping": {"status": "restored_order_approximate_time", "verified_pts": False,
                    "timestamp_precision_seconds": .01,
                    "timestamp_method": "legacy best_source_frame_idx / OpenCV CAP_PROP_FPS rounded to two decimals; recovered from keyframe filename",
                    "order_method": "numeric cached frame IDs aligned to lexicographically sorted producer keyframe paths and camera_poses rows",
                    "source_frame_indices_verified": False},
        "raw_depth_units": "relative", "raw_pose_translation_units": "relative",
        "metric_scale_factor": None, "pose_convention": "camera_to_world (legacy producer convention; scene axes not independently verified)",
        "confidence_convention": "raw model conf; range reported per frame, calibration and threshold semantics unverified",
        "viewer": {"path": str(viewer_dir.absolute()), "metadata": metadata, "assets": viewer_assets,
                   "scale_uncertainty_interpretation": "estimator uncertainty, not measured error",
                   "raw_to_viewer_scale_verified": False,
                   "postprocess": "may include yaw_rectify and carving; no raw-to-viewer equivalence asserted"},
        "model": model, "stream_configuration_files": configs,
        "known_stream_settings": stream_settings,
        "frames": frames,
        "limitations": ["Filename timestamps approximate frame-index/FPS time, not original presentation timestamps.",
                        "Source image association is an inference until extraction provenance is independently verified.",
                        "Absolute scale is unavailable; viewer units m do not establish a raw-depth scale factor.",
                        "Confidence calibration and scene axes remain unverified; use only as a visual historical baseline."]}
    # Reject non-finite viewer metadata before making output artifacts.
    json.dumps(manifest, allow_nan=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest["image_spotcheck"] = _review_images(keys, source_video, output_dir, spotcheck)
    write_json(output_dir / "legacy_manifest.json", manifest)
    review = ("# Historical DA3-Streaming baseline\n\n"
              f"Imported {len(frames)} cached frames, keyframes, and validated camera-to-world poses by producer order. "
              "Cached NPZ files were read with allow_pickle=False, one archive at a time. Sources were not modified.\n\n"
              "Filename times have 0.01-second precision and derive from selected frame index / legacy OpenCV FPS; "
              "they are approximate times, not verified PTS. Image spotchecks are inference only, not exact source-frame proof.\n\n"
              "Raw depth and camera translation remain in relative units. The viewer's units flag does not provide "
              "the missing absolute scale factor. Viewer scale_uncertainty is estimator uncertainty, not measured error. "
              "Viewer postprocessing may include yaw rectification and carving.\n\n"
              "Use as a visual legacy baseline until frame association, coordinate convention, confidence calibration, "
              "and metric scale are verified. Review the contact sheet when available; viewer artifacts remain external references.\n")
    with (output_dir / "baseline_review.md").open("x", encoding="utf-8") as stream:
        stream.write(review)
    return manifest
