#!/usr/bin/env python3
"""Audit a capture on CPU and export traceable portrait keyframes."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from room_reconstruction.ingestion import audit_video
from room_reconstruction.provenance import create_run_dir, git_snapshot, load_yaml, sha256_file, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", default="configs/capture_l2bp3.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    capture_path = ROOT / args.capture
    config = load_yaml(capture_path)
    run = create_run_dir(ROOT, args.run_id)
    try:
        write_json(run / "manifest.json", {"stage": "video_audit", "status": "started",
                   "source": git_snapshot(ROOT), "capture_config_sha256": sha256_file(capture_path)})
        report = audit_video(ROOT / config["input_video"], run,
                             target_fps=config["sampling"]["target_fps"],
                             max_keyframes=config["sampling"]["max_keyframes"],
                             expected_sha256=config["verified_metadata"]["sha256"], threads=args.threads,
                             sampling_mode=config["sampling"].get("mode", "uniform"),
                             adaptive_target_fps=config["sampling"].get("adaptive_target_fps"),
                             motion_threshold=config["sampling"].get("motion_threshold", .08))
        write_json(run / "status.json", {"status": "complete", "stage": "video_audit"})
        print(json.dumps({"run": str(run), "keyframes": report["keyframes"],
                          "decoded_frames": report["decoded_frames"],
                          "largest_temporal_gap_seconds": report["largest_temporal_gap_seconds"]}, indent=2))
        return 0
    except Exception as exc:
        write_json(run / "status.json", {"status": "failed", "stage": "video_audit", "reason": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
