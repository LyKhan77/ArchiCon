"""Video QA with presentation timestamps, bounded decoding and review artifacts."""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .provenance import sha256_file, write_json


def _json_command(args: list[str]) -> dict:
    result = subprocess.run(args, capture_output=True, text=True, timeout=600)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed: {result.stderr[-2000:]}")
    return json.loads(result.stdout)


def probe_video(video: Path) -> dict:
    payload = _json_command([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video),
    ])
    streams = [s for s in payload["streams"] if s.get("codec_type") == "video"]
    if len(streams) != 1:
        raise ValueError("Expected one video stream")
    stream = streams[0]
    width, height = int(stream["width"]), int(stream["height"])
    rotation = next((float(s["rotation"]) for s in stream.get("side_data_list", [])
                     if "rotation" in s), float(stream.get("tags", {}).get("rotate", 0)))
    if rotation % 90 != 0:
        raise ValueError("This audit supports right-angle display rotations only")
    display = [height, width] if int(rotation) % 180 else [width, height]
    return {"format": payload["format"], "stream": stream,
            "rotation_degrees": rotation, "display_size": display}


def parse_timestamps(payload: dict) -> tuple[float, list[float]]:
    frames = payload.get("frames", [])
    if not frames:
        raise ValueError("No presentation timestamps found")
    try:
        absolute = [float(f["best_effort_timestamp_time"]) for f in frames]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Every decoded frame needs a presentation timestamp") from exc
    if any(not math.isfinite(t) for t in absolute):
        raise ValueError("Non-finite presentation timestamp")
    if any(b <= a for a, b in zip(absolute, absolute[1:])):
        raise ValueError("Presentation timestamps must increase strictly")
    return absolute[0], [t - absolute[0] for t in absolute]


def probe_timestamps(video: Path) -> tuple[float, list[float]]:
    return parse_timestamps(_json_command([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(video),
    ]))


def choose_best_frames(rows: list[dict], target_fps: float, max_keyframes: int) -> list[dict]:
    if not math.isfinite(target_fps) or target_fps <= 0 or max_keyframes < 2:
        raise ValueError("Positive target_fps and max_keyframes >= 2 are required")
    windows: dict[int, dict] = {}
    for row in rows:
        key = math.floor((row["timestamp_seconds"] + 1e-9) * target_fps)
        if key not in windows or row["sharpness"] > windows[key]["sharpness"]:
            windows[key] = row
    selected = [windows[k] for k in sorted(windows)]
    if len(selected) > max_keyframes:
        indices = np.linspace(0, len(selected) - 1, max_keyframes).round().astype(int)
        selected = [selected[int(i)] for i in indices]
    return selected


def choose_adaptive_frames(rows: list[dict], base_fps: float, high_fps: float,
                           motion_threshold: float, max_keyframes: int) -> list[dict]:
    """Select original frames densely only in intervals with measured motion."""
    values = (base_fps, high_fps, motion_threshold)
    if (not all(math.isfinite(value) for value in values) or base_fps <= 0
            or high_fps < base_fps or not 0 <= motion_threshold <= 1
            or max_keyframes < 2):
        raise ValueError("Adaptive sampling requires valid rates, threshold and max_keyframes")
    windows: dict[tuple[float, int], dict] = {}
    for row in rows:
        motion = float(row.get("motion_score", 0.0))
        if not math.isfinite(motion) or not 0 <= motion <= 1:
            raise ValueError("Motion scores must be finite values in [0, 1]")
        rate = high_fps if motion >= motion_threshold else base_fps
        key = (rate, math.floor((row["timestamp_seconds"] + 1e-9) * rate))
        if key not in windows or row["sharpness"] > windows[key]["sharpness"]:
            windows[key] = row
    selected = sorted(windows.values(), key=lambda row: (row["timestamp_seconds"], row["source_index"]))
    if len(selected) > max_keyframes:
        indices = np.linspace(0, len(selected) - 1, max_keyframes).round().astype(int)
        selected = [selected[int(index)] for index in indices]
    return selected


def assign_splits(times: list[float], seed: int, validation_fraction: float,
                  block_seconds: float = 2.0) -> list[str]:
    if not 0 < validation_fraction < 1 or block_seconds <= 0:
        raise ValueError("Invalid split configuration")
    blocks = sorted({math.floor(t / block_seconds) for t in times})
    shuffled = blocks.copy()
    random.Random(seed).shuffle(shuffled)
    count = min(len(blocks) - 1, max(1, round(len(blocks) * validation_fraction))) if len(blocks) > 1 else 0
    validation = set(shuffled[:count])
    return ["validation" if math.floor(t / block_seconds) in validation else "train" for t in times]


def _thumbnail_dimensions(display: list[int], longest_edge: int) -> tuple[int, int]:
    if longest_edge < 32:
        raise ValueError("review_longest_edge must be at least 32")
    factor = min(1.0, longest_edge / max(display))
    return max(2, int(display[0] * factor) // 2 * 2), max(2, int(display[1] * factor) // 2 * 2)


def _scan(video: Path, run: Path, times: list[float], display: list[int],
          target_fps: float, longest_edge: int, threads: int) -> list[dict]:
    width, height = _thumbnail_dimensions(display, longest_edge)
    size = width * height * 3
    thumbnails = run / "review/thumbnails"
    thumbnails.mkdir(parents=True)
    rows = []
    best_row = best_image = None
    previous_window = None
    previous_gray = None

    def save_best() -> None:
        if best_row is not None:
            if not cv2.imwrite(str(run / best_row["thumbnail_path"]), best_image):
                raise RuntimeError("Cannot write review thumbnail")

    args = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", str(threads),
            "-i", str(video), "-map", "0:v:0", "-filter_threads", "1", "-vf", f"scale={width}:{height}",
            "-fps_mode", "passthrough", "-pix_fmt", "bgr24", "-threads", "1", "-f", "rawvideo", "pipe:1"]
    with tempfile.TemporaryFile() as error_log, (run / "frame_quality.jsonl").open("x") as quality:
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=error_log)
        try:
            index = 0
            while True:
                raw = process.stdout.read(size)
                if not raw:
                    break
                if len(raw) != size or index >= len(times):
                    raise ValueError("Decoder frame count or dimensions disagree with ffprobe")
                image = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                motion_score = 0.0 if previous_gray is None else float(
                    cv2.absdiff(gray, previous_gray).mean() / 255.0)
                row = {"source_index": index, "timestamp_seconds": times[index],
                       "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                       "mean_luma": float(gray.mean()), "dark_fraction": float((gray < 8).mean()),
                       "bright_fraction": float((gray > 247).mean()),
                       "motion_score": motion_score,
                       "thumbnail_path": f"review/thumbnails/frame_{index:06d}.jpg"}
                quality.write(json.dumps(row, allow_nan=False) + "\n")
                rows.append(row)
                window = math.floor((times[index] + 1e-9) * target_fps)
                if window != previous_window:
                    save_best()
                    best_row, best_image = row, image.copy()
                    previous_window = window
                elif row["sharpness"] > best_row["sharpness"]:
                    best_row, best_image = row, image.copy()
                previous_gray = gray
                index += 1
                if index % 1500 == 0:
                    print(f"Audited {index}/{len(times)} source frames", file=sys.stderr, flush=True)
            save_best()
            returncode = process.wait(timeout=60)
            if returncode:
                error_log.seek(0)
                raise RuntimeError(error_log.read().decode(errors="replace")[-2000:])
            if index != len(times):
                raise ValueError(f"Decoded {index} frames but ffprobe reported {len(times)}")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()
    return rows


def _contact_sheets(run: Path, rows: list[dict], prefix: str) -> list[str]:
    outputs = []
    width, height, label = 180, 320, 28
    for offset in range(0, len(rows), 12):
        canvas = np.full((3 * (height + label), 4 * width, 3), 245, np.uint8)
        for cell, row in enumerate(rows[offset:offset + 12]):
            image = cv2.imread(str(run / row["thumbnail_path"]))
            if image is None:
                raise ValueError("Missing review thumbnail")
            factor = min(width / image.shape[1], height / image.shape[0])
            resized = cv2.resize(image, (round(image.shape[1] * factor), round(image.shape[0] * factor)))
            x, y = (cell % 4) * width, (cell // 4) * (height + label)
            canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
            text = f"{row['timestamp_seconds']:.3f}s  f{row['source_index']}"
            cv2.putText(canvas, text, (x + 4, y + height + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)
        relative = f"review/{prefix}_{offset // 12 + 1:02d}.jpg"
        if not cv2.imwrite(str(run / relative), canvas):
            raise RuntimeError("Cannot write contact sheet")
        outputs.append(relative)
    return outputs


def _extract_frames(video: Path, run: Path, selected: list[dict], threads: int) -> None:
    if not selected:
        raise ValueError("At least one source frame must be selected")

    def balanced_sum(terms: list[str]) -> str:
        if len(terms) == 1:
            return terms[0]
        middle = len(terms) // 2
        return f"({balanced_sum(terms[:middle])})+({balanced_sum(terms[middle:])})"

    frames = run / "frames"
    frames.mkdir()
    # A linear sum of hundreds of terms exceeds FFmpeg expression parser depth.
    expression = balanced_sum([f"eq(n\\,{row['source_index']})" for row in selected])
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", str(threads),
               "-i", str(video), "-map", "0:v:0", "-filter_threads", "1", "-vf", f"select={expression}",
               "-fps_mode", "passthrough", "-q:v", "2", "-threads", str(threads),
               "-start_number", "0", str(frames / "keyframe_%06d.jpg")]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if result.returncode:
        logs = run / "logs"
        logs.mkdir(exist_ok=True)
        (logs / "extraction_error.log").write_text(result.stderr)
        raise RuntimeError(f"Keyframe extraction failed; inspect {logs / 'extraction_error.log'}")
    if len(list(frames.glob("*.jpg"))) != len(selected):
        raise ValueError("Exported keyframes do not match selected source indices")
    for i, row in enumerate(selected):
        row["image_path"] = f"frames/keyframe_{i:06d}.jpg"


def audit_video(video: Path, run_dir: Path, target_fps: float = 2.0, max_keyframes: int = 320,
                review_longest_edge: int = 512, split_seed: int = 7, validation_fraction: float = 0.2,
                expected_sha256: str | None = None, threads: int = 4,
                sampling_mode: str = "uniform", adaptive_target_fps: float | None = None,
                motion_threshold: float = 0.03) -> dict:
    video, run = Path(video).resolve(), Path(run_dir)
    if not math.isfinite(target_fps) or target_fps <= 0 or max_keyframes < 2 or threads < 1:
        raise ValueError("Invalid audit parameters")
    cv2.setNumThreads(threads)
    digest = sha256_file(video)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Source video checksum differs from capture configuration")
    metadata = probe_video(video)
    origin, times = probe_timestamps(video)
    rows = _scan(video, run, times, metadata["display_size"], target_fps, review_longest_edge, threads)
    if sampling_mode not in ("uniform", "adaptive"):
        raise ValueError("sampling_mode must be uniform or adaptive")
    if sampling_mode == "adaptive":
        high_fps = target_fps if adaptive_target_fps is None else adaptive_target_fps
        selected = [dict(row) for row in choose_adaptive_frames(
            rows, target_fps, high_fps, motion_threshold, max_keyframes)]
    else:
        high_fps = None
        selected = [dict(row) for row in choose_best_frames(rows, target_fps, max_keyframes)]
    if not selected:
        raise ValueError("No keyframes selected")
    median = float(np.median([row["sharpness"] for row in selected]))
    splits = assign_splits([r["timestamp_seconds"] for r in selected], split_seed, validation_fraction)
    for row, split in zip(selected, splits):
        row.update({"frame_id": f"source_{row['source_index']:06d}", "source_pts_seconds": origin + row["timestamp_seconds"],
                    "split": split, "low_sharpness_relative": row["sharpness"] < 0.35 * median,
                    "decision": "retained_for_coverage"})
    _extract_frames(video, run, selected, threads)
    for row in selected:
        path = run / row["image_path"]
        image = cv2.imread(str(path))
        if image is None or [image.shape[1], image.shape[0]] != metadata["display_size"]:
            raise ValueError("Exported image orientation differs from display metadata")
        row["image_sha256"] = sha256_file(path)
        row["display_size"] = metadata["display_size"]
    overview = [selected[int(i)] for i in np.linspace(0, len(selected) - 1, min(24, len(selected))).round()]
    contacts = _contact_sheets(run, overview, "overview")
    contacts += _contact_sheets(run, [r for r in selected if 15 <= r["timestamp_seconds"] < 45], "pilot")
    contacts += _contact_sheets(run, [r for r in selected if 110 <= r["timestamp_seconds"] < 140], "stress")
    duration = float(metadata["format"]["duration"])
    selected_times = [r["timestamp_seconds"] for r in selected]
    gaps = [b - a for a, b in zip([0.0, *selected_times], [*selected_times, duration])]
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_video": str(video), "source_sha256": digest, "duration_seconds": duration,
        "encoded_size": [metadata["stream"]["width"], metadata["stream"]["height"]],
        "display_size": metadata["display_size"], "rotation_degrees": metadata["rotation_degrees"],
        "decoded_frames": len(times), "keyframes": len(selected), "timestamp_basis": "decoder_best_effort_pts",
        "source_pts_origin_seconds": origin, "first_selected_seconds": selected_times[0],
        "last_selected_seconds": selected_times[-1], "largest_temporal_gap_seconds": max(gaps),
        "target_fps": target_fps, "sampling_mode": sampling_mode,
        "adaptive_target_fps": high_fps, "motion_threshold": motion_threshold,
        "high_motion_source_frames": sum(row["motion_score"] >= motion_threshold for row in rows),
        "selected_high_motion_frames": sum(row["motion_score"] >= motion_threshold for row in selected),
        "review_longest_edge": review_longest_edge,
        "selected_sharpness_median": median,
        "low_sharpness_flagged": sum(r["low_sharpness_relative"] for r in selected),
        "train_frames": splits.count("train"), "validation_frames": splits.count("validation"),
        "split_seed": split_seed, "split_block_seconds": 2.0, "contact_sheets": contacts,
        "limitations": ["Sharpness/exposure are proxies, not semantic blur or reflection labels.",
                        "Low-texture windows retained; masks and geometric consistency checks remain pending.",
                        "No reconstruction or verified scale is produced by this audit."],
    }
    write_json(run / "source_manifest.json", {"source_sha256": digest, "source_path": str(video), "metadata": metadata})
    write_json(run / "frame_manifest.json", {"schema_version": 1, "source_sha256": digest,
               "timestamp_basis": report["timestamp_basis"], "frames": selected})
    write_json(run / "audit_report.json", report)
    reports = run / "reports"
    reports.mkdir()
    lines = ["# Video audit", "", f"Source: `{video.name}`", f"SHA-256: `{digest}`", "",
             f"- Duration: {duration:.3f} seconds; decoded frames: {len(times)}.",
             f"- Display: {metadata['display_size']}; source rotation: {metadata['rotation_degrees']} degrees.",
             f"- Selected: {len(selected)} full-resolution JPEG frames, train {splits.count('train')}, validation {splits.count('validation')}.",
             f"- Largest temporal gap including endpoints: {max(gaps):.3f} seconds.",
             f"- Low relative sharpness flagged: {report['low_sharpness_flagged']} (retained).", "",
             "## Contact sheets", "", *[f"- [{Path(p).name}](../{p})" for p in contacts], "",
             "## Limitations", "", *[f"- {s}" for s in report["limitations"]], ""]
    (reports / "video_audit.md").write_text("\n".join(lines))
    return report
