import hashlib
import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import cv2

from room_reconstruction.ingestion import (_extract_frames, assign_splits, audit_video,
                                           choose_adaptive_frames, choose_best_frames, parse_timestamps)


class IngestionTests(unittest.TestCase):
    def test_audit_cli_persists_failure_when_provenance_collection_fails(self):
        script = Path(__file__).resolve().parents[1] / "scripts/audit_video.py"
        spec = importlib.util.spec_from_file_location("audit_cli_test", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "configs").mkdir()
            (root / "configs/capture.yaml").write_text("schema_version: 1\n")
            with patch.object(module, "ROOT", root), patch.object(module, "git_snapshot", side_effect=RuntimeError("provenance failed")), patch(
                "sys.argv", [str(script), "--capture", "configs/capture.yaml", "--run-id", "failure-case"]
            ), self.assertRaisesRegex(RuntimeError, "provenance failed"):
                module.main()
            status = json.loads((root / "outputs/failure-case/status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertIn("provenance failed", status["reason"])

    def test_timestamps_use_pts_and_preserve_nonuniform_intervals(self):
        origin, times = parse_timestamps({"frames": [
            {"best_effort_timestamp_time": "5.200"},
            {"best_effort_timestamp_time": "5.250"},
            {"best_effort_timestamp_time": "6.000"},
        ]})
        self.assertEqual(origin, 5.2)
        self.assertAlmostEqual(times[1], 0.05)
        self.assertAlmostEqual(times[2], 0.8)

    def test_missing_duplicate_or_nonfinite_pts_fail_closed(self):
        for values in [[{}, {}], [{"best_effort_timestamp_time": "nan"}],
                       [{"best_effort_timestamp_time": "1"}] * 2]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                parse_timestamps({"frames": values})

    def test_best_per_window_retains_low_texture_windows(self):
        rows = [dict(source_index=i, timestamp_seconds=t, sharpness=s) for i, (t, s) in enumerate(
            [(0.0, 2), (0.2, 5), (0.5, 0), (0.7, 0.1), (1.0, 9)]
        )]
        chosen = choose_best_frames(rows, target_fps=2, max_keyframes=10)
        self.assertEqual([r["source_index"] for r in chosen], [1, 3, 4])
        self.assertEqual(len(choose_best_frames(rows, 2, 2)), 2)

    def test_adaptive_selection_adds_original_frames_in_motion_windows(self):
        rows = [dict(source_index=i, timestamp_seconds=i * .125, sharpness=10 + i,
                     motion_score=.9 if 1.0 <= i * .125 < 1.5 else .0)
                for i in range(16)]
        chosen = choose_adaptive_frames(rows, base_fps=2, high_fps=4, motion_threshold=.5,
                                        max_keyframes=30)
        self.assertGreater(len(chosen), len(choose_best_frames(rows, 2, 30)))
        self.assertEqual([row['source_index'] for row in chosen], sorted({row['source_index'] for row in chosen}))
        self.assertTrue(all(row in rows for row in chosen))

    def test_adaptive_selection_is_bounded_and_validates_rates(self):
        rows = [dict(source_index=i, timestamp_seconds=i / 10, sharpness=1, motion_score=.9)
                for i in range(100)]
        self.assertEqual(len(choose_adaptive_frames(rows, 2, 4, .5, 12)), 12)
        for args in [(0, 4, .5, 12), (2, 1, .5, 12), (2, 4, -0.1, 12)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                choose_adaptive_frames(rows, *args)

    def test_temporal_split_is_deterministic_and_never_splits_a_block(self):
        times = [i * 0.5 for i in range(40)]
        first = assign_splits(times, seed=7, validation_fraction=0.2)
        self.assertEqual(first, assign_splits(times, seed=7, validation_fraction=0.2))
        self.assertIn("train", first)
        self.assertIn("validation", first)
        for block in range(10):
            self.assertEqual(len(set(first[block * 4:(block + 1) * 4])), 1)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_extracts_hundreds_of_irregular_source_indices(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "many.mp4"
            subprocess.run([
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                "-i", "color=size=32x24:rate=604:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
            ], check=True)
            run = root / "audit"
            run.mkdir()
            selected = [{"source_index": i} for i in range(0, 604, 2)]
            _extract_frames(video, run, selected, threads=1)
            self.assertEqual(len(list((run / "frames").glob("*.jpg"))), 302)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_real_rotated_vfr_video_audit_preserves_source_and_orientation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, video = root / "base.mov", root / "portrait.mov"
            subprocess.run([
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                "-i", "testsrc=size=64x48:rate=6:duration=1", "-vf", "select=not(eq(n\\,2))",
                "-fps_mode", "vfr", "-c:v", "libx264", "-bf", "0", "-pix_fmt", "yuv420p", str(base),
            ], check=True)
            subprocess.run([
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-display_rotation", "90", "-i", str(base),
                "-c", "copy", str(video),
            ], check=True)
            before = hashlib.sha256(video.read_bytes()).hexdigest()
            run = root / "audit"
            run.mkdir()
            report = audit_video(video, run, review_longest_edge=64, threads=1)
            manifest = json.loads((run / "frame_manifest.json").read_text())
            self.assertEqual(report["decoded_frames"], 5)
            self.assertEqual(report["display_size"], [48, 64])
            self.assertEqual(hashlib.sha256(video.read_bytes()).hexdigest(), before)
            image = cv2.imread(str(run / manifest["frames"][0]["image_path"]))
            self.assertEqual(image.shape[:2], (64, 48))
            pts = [r["timestamp_seconds"] for r in manifest["frames"]]
            self.assertEqual(pts, sorted(set(pts)))
            self.assertTrue((run / "reports/video_audit.md").is_file())
            self.assertTrue((run / "review/overview_01.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
