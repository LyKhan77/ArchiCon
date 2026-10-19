#!/usr/bin/env python3
"""Import historical DA3 cached results into a new, non-overwriting review run."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from room_reconstruction.legacy import import_legacy
from room_reconstruction.provenance import (
    create_run_dir, git_snapshot, load_yaml, sha256_file, write_json,
)


def main() -> int:
    checkout_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/paths.server.yaml"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--keyframes", type=Path, help="Defaults to legacy_project/data/keyframes/l2bp3_stream")
    parser.add_argument("--skip-image-check", action="store_true")
    args = parser.parse_args()
    run = None
    try:
        if not args.config.is_absolute():
            args.config = checkout_root / args.config
        config = load_yaml(args.config)
        paths = config["paths"]
        root = Path(paths["project_root"]).resolve()
        if root != checkout_root:
            raise ValueError(f"paths.project_root must match this checkout: {checkout_root}")
        import cv2
        requested_threads = config.get("compute", {}).get("cpu_threads", 4)
        if isinstance(requested_threads, bool) or not isinstance(requested_threads, int) or requested_threads < 1:
            raise ValueError("compute.cpu_threads must be a positive integer")
        cv2.setNumThreads(min(4, requested_threads))
        if cv2.getNumThreads() > min(4, requested_threads):
            cv2.setNumThreads(0)

        def resolve(name: str) -> Path:
            value = Path(paths[name])
            return value if value.is_absolute() else root / value

        keyframes = args.keyframes or resolve("legacy_project") / "data/keyframes/l2bp3_stream"
        # Enforce read-only boundaries before create_run_dir can make a directory.
        prospective = (root / "outputs" / args.run_id).resolve()
        for source in (resolve("legacy_stream"), keyframes, resolve("legacy_viewer")):
            if prospective.is_relative_to(source.resolve()) or source.resolve().is_relative_to(prospective):
                raise ValueError("Output must be separate from read-only legacy source directories")
        if resolve("input_video").resolve().is_relative_to(prospective):
            raise ValueError("Output must be separate from source video")
        run = create_run_dir(root, args.run_id)
        manifest = import_legacy(resolve("legacy_stream"), keyframes, resolve("legacy_viewer"),
                                 resolve("input_video"), run, spotcheck=not args.skip_image_check)
        write_json(run / "run_provenance.json", {
            "command": sys.argv, "config_path": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config), "config_snapshot": config,
            "git": git_snapshot(Path(__file__).resolve().parents[1]),
            "python": sys.version,
        })
        print(f"Imported {manifest['frame_count']} frames: {run / 'legacy_manifest.json'}")
        print("Baseline remains visual-only; raw scale is relative and filename times are approximate.")
        write_json(run / "status.json", {"status": "complete", "stage": "legacy_import",
                                         "frame_count": manifest["frame_count"]})
    except Exception as exc:
        if run is not None:
            try:
                write_json(run / "status.json", {"status": "failed", "stage": "legacy_import",
                                                 "reason": str(exc), "error_type": type(exc).__name__})
            except Exception as status_exc:
                parser.exit(1, f"Legacy import failed: {exc}; cannot persist failure status: {status_exc}\n")
        parser.exit(1, f"Legacy import failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
