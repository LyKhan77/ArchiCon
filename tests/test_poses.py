import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from room_reconstruction import poses


class PoseTests(unittest.TestCase):
    def test_pilot_keeps_train_only_half_open_interval_and_caps_evenly(self):
        rows = [dict(timestamp_seconds=i, split='validation' if i % 7 == 0 else 'train', source_index=i)
                for i in range(100)]
        result = poses.select_pilot_frames(rows, 15, 90)
        self.assertEqual(len(result), 55)
        self.assertEqual(result[0]['source_index'], 15)
        self.assertEqual(result[-1]['source_index'], 89)
        self.assertTrue(all(r['split'] == 'train' and 15 <= r['timestamp_seconds'] < 90 for r in result))

    def test_quality_gate_uses_observation_residuals_and_registration(self):
        summary = poses.quality_summary([0, 1, 1, 2], 9, 10)
        self.assertTrue(summary['passed'])
        self.assertEqual(summary['median_native_pixels'], 1)
        self.assertFalse(poses.quality_summary([0], 8, 10)['passed'])
        self.assertFalse(poses.quality_summary([], 10, 10)['passed'])
        self.assertFalse(poses.quality_summary([float('nan')], 10, 10)['passed'])
        self.assertFalse(poses.quality_summary([0] * 9 + [100], 10, 10)['passed'])

    def test_canonical_intrinsics_match_native_size_and_invert_pose(self):
        camera = SimpleNamespace(width=1080, height=1920, model_name='OPENCV',
            params=np.array([800, 900, 540, 960, 0, 0, 0, 0]),
            calibration_matrix=lambda: np.array([[800, 0, 540], [0, 900, 960], [0, 0, 1]]))
        image = SimpleNamespace(name='image.jpg', camera_id=1,
            cam_from_world=lambda: SimpleNamespace(matrix=lambda: np.column_stack([np.eye(3), [1, 2, 3]])))
        model = SimpleNamespace(reg_image_ids=lambda: [1], images={1: image}, cameras={1: camera})
        row = dict(image_path='frames/image.jpg', frame_id='source_01', source_index=1,
            timestamp_seconds=15.5, display_size=[1080, 1920], image_sha256='hash')
        result = poses.canonical_cameras(model, [row])['frames'][0]
        np.testing.assert_allclose(np.array(result['c2w'])[:3, 3], [-1, -2, -3])
        self.assertEqual(result['K'][0][2], 540)
        row['display_size'] = [540, 960]
        with self.assertRaisesRegex(ValueError, 'native'):
            poses.canonical_cameras(model, [row])

    def test_copy_checks_hash_and_leaves_audit_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit, images = root / 'audit', root / 'images'
            audit.mkdir()
            (audit / 'image.jpg').write_bytes(b'native bytes')
            row = dict(image_path='image.jpg', image_sha256='wrong')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                poses.copy_pilot_images(audit, images, [row])
            self.assertEqual((audit / 'image.jpg').read_bytes(), b'native bytes')

    def test_copy_rejects_paths_outside_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / 'audit'
            audit.mkdir()
            with self.assertRaises(ValueError):
                poses.copy_pilot_images(audit, root / 'images', [dict(image_path='../secret', image_sha256='')])

    def test_actual_camera_model_projection_drives_residual(self):
        class Camera:
            def img_from_cam(self, xyz):
                # Nonlinear model: pinhole-only residual would be wrong.
                return np.array([xyz[0] / xyz[2] + 2, xyz[1] / xyz[2]])
        class Transform:
            def matrix(self):
                return np.column_stack([np.eye(3), np.zeros(3)])
        class Observation:
            point3D_id = 7
            xy = np.array([3, 0.5])
            def has_point3D(self):
                return True
        class Image:
            camera_id = 1
            points2D = [Observation()]
            def cam_from_world(self):
                return Transform()
        class Point:
            xyz = np.array([2, 1, 2])
        class Model:
            images = {4: Image()}
            cameras = {1: Camera()}
            points3D = {7: Point()}
            def reg_image_ids(self):
                return [4]
        self.assertEqual(poses.observation_residuals(Model()), [0.0])

    def test_cli_failure_after_creation_records_failed_status(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/reconstruct_cameras.py'
        spec = importlib.util.spec_from_file_location('camera_cli', path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory() as tmp, patch.object(cli, 'ROOT', Path(tmp)), patch.object(cli, 'run_pilot', side_effect=RuntimeError('unexpected')):
            self.assertEqual(cli.main(['--audit-run', 'audit', '--run-id', 'pilot']), 2)
            state = json.loads((Path(tmp) / 'outputs/pilot/status.json').read_text())
            self.assertEqual(state['status'], 'failed')
            self.assertIn('unexpected', state['reason'])

    def test_resource_rejection_stops_before_backend_and_frame_copy(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/reconstruct_cameras.py'
        spec = importlib.util.spec_from_file_location('camera_cli_resources', path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'configs').mkdir()
            (root / 'configs/paths.server.yaml').write_text(
                f'profile: server\npaths:\n  project_root: {root}\ncompute: {{}}\n')
            with patch.object(cli, 'ROOT', root), patch.object(cli, 'parse_mem_available', return_value=0):
                self.assertEqual(cli.main(['--audit-run', 'audit', '--run-id', 'pilot']), 2)
            run = root / 'outputs/pilot'
            self.assertFalse((run / 'images').exists())
            self.assertFalse(json.loads((run / 'resources.json').read_text())['resource_ready'])
            self.assertEqual(json.loads((run / 'status.json').read_text())['status'], 'failed')

    def test_upstream_audit_must_be_complete_before_manifest_selection(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/reconstruct_cameras.py'
        spec = importlib.util.spec_from_file_location('camera_cli_audit_gate', path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        statuses = [None, '{', 'null', '[]', '{}',
            '{"stage":"video_audit","status":"failed"}',
            '{"stage":"video_audit","status":"started"}',
            '{"stage":"camera_pilot","status":"complete"}',
            '{"status":"complete"}']
        for state in statuses:
            with self.subTest(state=state), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                audit, run = root / 'outputs/audit', root / 'outputs/pilot'
                audit.mkdir(parents=True)
                run.mkdir()
                (root / 'config.yaml').write_text(f'profile: server\npaths:\n  project_root: {root}\n')
                (audit / 'frame_manifest.json').write_text('{"frames":[]}')
                if state is not None:
                    (audit / 'status.json').write_text(state)
                args = SimpleNamespace(config='config.yaml', threads=4, audit_run='audit', start=15, end=45)
                with patch.object(cli, 'evaluate_resources', return_value={'resource_ready': True}), \
                        patch.object(poses, 'select_pilot_frames', side_effect=AssertionError('Manifest selection reached before audit gate')) as select, \
                        patch.object(poses, 'copy_pilot_images') as copy:
                    with self.assertRaisesRegex(ValueError, 'completed video_audit'):
                        cli.run_pilot(root, run, args)
                    select.assert_not_called()
                    copy.assert_not_called()

    def test_completed_video_audit_allows_manifest_selection(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/reconstruct_cameras.py'
        spec = importlib.util.spec_from_file_location('camera_cli_audit_complete', path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            audit, run = root / 'outputs/audit', root / 'outputs/pilot'
            audit.mkdir(parents=True)
            run.mkdir()
            (root / 'config.yaml').write_text(f'profile: server\npaths:\n  project_root: {root}\n')
            (audit / 'frame_manifest.json').write_text('{"frames":[]}')
            (audit / 'status.json').write_text('{"stage":"video_audit","status":"complete"}')
            args = SimpleNamespace(config='config.yaml', threads=4, audit_run='audit', start=15, end=45)
            with patch.object(cli, 'evaluate_resources', return_value={'resource_ready': True}), \
                    patch.object(poses, 'select_pilot_frames', side_effect=RuntimeError('selection allowed')) as select:
                with self.assertRaisesRegex(RuntimeError, 'selection allowed'):
                    cli.run_pilot(root, run, args)
                select.assert_called_once()


if __name__ == '__main__':
    unittest.main()
