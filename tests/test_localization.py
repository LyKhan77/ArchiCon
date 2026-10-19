import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from room_reconstruction import localization as loc


def load_cli():
    path = Path(__file__).resolve().parents[1] / 'scripts/localize_validation.py'
    spec = importlib.util.spec_from_file_location('localization_cli', path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


class LocalizationTests(unittest.TestCase):
    def test_select_exactly_eight_validation_frames_half_open_interval(self):
        rows = [dict(split='validation' if i % 2 else 'train', timestamp_seconds=i,
                     image_path=f'frames/{i}.jpg', frame_id=str(i)) for i in range(60)]
        selected = loc.select_validation_frames(rows)
        self.assertEqual(len(selected), 8)
        self.assertTrue(all(r['split'] == 'validation' and 15 <= r['timestamp_seconds'] < 45 for r in selected))
        self.assertEqual(selected[0]['timestamp_seconds'], 15)
        self.assertEqual(selected[-1]['timestamp_seconds'], 43)
        with self.assertRaisesRegex(ValueError, 'eight'):
            loc.select_validation_frames(rows[:20])

    def test_upstream_gates_reject_incomplete_and_unrelated_pilot(self):
        audit = dict(stage='video_audit', status='complete')
        camera = dict(status='camera_passed', camera_ready=True)
        manifest = dict(audit_run='audit', audit_manifest_sha256='hash')
        loc.validate_upstream(audit, camera, manifest, 'audit', 'hash')
        for bad in [{}, None, [], dict(stage='video_audit', status='failed')]:
            with self.assertRaisesRegex(ValueError, 'completed video_audit'):
                loc.validate_upstream(bad, camera, manifest, 'audit', 'hash')
        with self.assertRaisesRegex(ValueError, 'camera_passed'):
            loc.validate_upstream(audit, dict(status='camera_rejected'), manifest, 'audit', 'hash')
        with self.assertRaisesRegex(ValueError, 'audit manifest'):
            loc.validate_upstream(audit, camera, manifest, 'audit', 'changed')

    def test_training_map_never_contains_validation_names(self):
        rows = [dict(split='train', image_path='frames/train.jpg'),
                dict(split='validation', image_path='frames/query.jpg')]
        loc.validate_training_names(['train.jpg'], rows)
        for names in [['query.jpg'], ['unknown.jpg'], ['train.jpg', 'query.jpg']]:
            with self.assertRaisesRegex(ValueError, 'training'):
                loc.validate_training_names(names, rows)

    def test_correspondences_discard_ambiguous_and_duplicate_world_points(self):
        keypoints = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        xyz = {10: np.array([1, 2, 3]), 11: np.array([2, 3, 4]), 12: np.array([3, 4, 5])}
        # Query 1 ties between tracks; query 2 loses duplicate track to query 0.
        votes = [(0, 10), (0, 10), (1, 11), (1, 12), (2, 10), (3, 12)]
        pixels, points, indices = loc.resolve_correspondences(keypoints, votes, xyz)
        np.testing.assert_array_equal(pixels, [[1, 2], [7, 8]])
        np.testing.assert_array_equal(points, [xyz[10], xyz[12]])
        self.assertEqual(indices, [(0, 10), (3, 12)])

    def test_residuals_use_distortion_and_reject_negative_depth(self):
        camera = SimpleNamespace(img_from_cam=lambda xyz: xyz[:2] / xyz[2] + [2, 0])
        transform = np.column_stack([np.eye(3), [0, 0, 0]])
        errors = loc.pose_residuals([[3, .5]], [[2, 1, 2]], camera, transform)
        self.assertEqual(errors, [0])
        with self.assertRaisesRegex(ValueError, 'front'):
            loc.pose_residuals([[0, 0]], [[1, 2, -1]], camera, transform)

    def test_localization_quality_gates_all_eight_and_bad_residuals(self):
        good = loc.query_quality([1] * 30, 40)
        self.assertTrue(good['passed'])
        self.assertFalse(loc.query_quality([1] * 29, 40)['passed'])
        self.assertFalse(loc.query_quality([1] * 30, 200)['passed'])
        self.assertFalse(loc.query_quality([1] * 28 + [8, 8], 40)['passed'])
        self.assertFalse(loc.query_quality([float('nan')] * 30, 40)['passed'])
        self.assertFalse(loc.query_quality([], 40)['passed'])
        self.assertTrue(loc.localization_quality([good] * 8)['passed'])
        self.assertFalse(loc.localization_quality([good] * 7)['passed'])

    def test_map_snapshot_detects_model_and_database_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'primary').mkdir()
            (root / 'primary/cameras.bin').write_bytes(b'fixed camera')
            (root / 'database.db').write_bytes(b'fixed db')
            before = loc.map_snapshot(root)
            self.assertEqual(before, loc.map_snapshot(root))
            (root / 'primary/cameras.bin').write_bytes(b'mutated camera')
            self.assertNotEqual(before, loc.map_snapshot(root))

    def test_query_undistortion_caps_actual_opencv_threads(self):
        import cv2
        original = cv2.getNumThreads()
        try:
            for threads in (1, 2, 4):
                with self.subTest(threads=threads), tempfile.TemporaryDirectory() as tmp:
                    cv2.setNumThreads(12)
                    result = loc.undistort_queries(Path(tmp), Path(tmp), {'frames': []}, threads=threads)
                    self.assertFalse(result['available'])
                    self.assertLessEqual(cv2.getNumThreads(), threads)
        finally:
            cv2.setNumThreads(original)

    @unittest.skipUnless(importlib.util.find_spec('pycolmap'), 'PyCOLMAP integration runs on server')
    def test_synthetic_fixed_camera_recovers_pose_with_outliers(self):
        import pycolmap as p
        rng = np.random.default_rng(23)
        xyz = rng.uniform([-2, -2, 4], [2, 2, 10], (100, 3))
        camera = p.Camera(model='OPENCV', width=1080, height=1920,
                          params=[900, 890, 540, 960, .04, -.02, .001, -.001])
        pixels = np.array([camera.img_from_cam(point) for point in xyz])
        pixels[:15] = rng.uniform([0, 0], [1080, 1920], (15, 2))
        original = camera.params.copy()
        result = loc.estimate_query_pose(pixels, xyz, camera, p)
        self.assertTrue(result['quality']['passed'])
        self.assertGreaterEqual(result['quality']['inlier_count'], 85)
        np.testing.assert_allclose(result['c2w'], np.eye(4), atol=1e-5)
        np.testing.assert_array_equal(camera.params, original)

    def test_cli_resource_gate_stops_before_copy(self):
        cli = load_cli()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'config.yaml').write_text(f'profile: server\npaths:\n  project_root: {root}\n')
            with patch.object(cli, 'ROOT', root), patch.object(cli, 'parse_mem_available', return_value=0):
                result = cli.main(['--audit-run', 'audit', '--camera-run', 'camera',
                                   '--run-id', 'query', '--config', 'config.yaml'])
            run = root / 'outputs/query'
            self.assertEqual(result, 2)
            self.assertFalse((run / 'images').exists())
            self.assertEqual(json.loads((run / 'status.json').read_text())['status'], 'failed')

    def test_cli_failed_status_no_overwrite(self):
        cli = load_cli()
        with tempfile.TemporaryDirectory() as tmp, patch.object(cli, 'ROOT', Path(tmp)), \
                patch.object(cli, 'run_localization', side_effect=RuntimeError('backend failure')):
            argv = ['--audit-run', 'audit', '--camera-run', 'camera', '--run-id', 'query']
            self.assertEqual(cli.main(argv), 2)
            path = Path(tmp) / 'outputs/query/status.json'
            original = path.read_bytes()
            self.assertEqual(cli.main(argv), 2)
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
