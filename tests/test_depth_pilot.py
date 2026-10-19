import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from room_reconstruction import depth_pilot as pilot
from room_reconstruction.provenance import sha256_file


class DepthPilotTests(unittest.TestCase):
    def fixture(self, root):
        camera = root / 'camera'
        images = camera / 'undistorted/images'
        images.mkdir(parents=True)
        rows = []
        for i in range(53):
            image = images / f'f{i}.jpg'
            image.write_bytes(str(i).encode())
            pose = np.eye(4)
            pose[:3, 3] = [i / 10, np.sin(i / 10), 0]
            rows.append(dict(frame_id=f'f{i}', image_name=image.name, image_sha256=sha256_file(image),
                timestamp_seconds=15 + i * 30 / 51, source_index=i, split='train' if i < 52 else 'validation',
                image_size=[1078, 1927], camera_model='PINHOLE',
                K=[[900, 0, 539], [0, 900, 963.5], [0, 0, 1]], c2w=pose.tolist()))
        for name, data in [('status', dict(status='camera_passed', camera_ready=True)),
                ('report', dict(stage='camera_pilot', quality=dict(passed=True), undistorted=True)),
                ('manifest', dict(stage='camera_pilot', selected_frames=rows))]:
            (camera / f'{name}.json').write_text(json.dumps(data))
        (camera / 'undistorted/cameras.json').write_text(json.dumps(dict(schema_version=1,
            pose_convention='OpenCV c2w; +x right, +y down, +z forward',
            world='Relative, unscaled COLMAP world', image_root='images', frames=rows[::-1])))
        return camera

    def test_all_52_train_frames_have_temporal_order_and_global_inverse_poses(self):
        with TemporaryDirectory() as temp:
            inputs = pilot.prepare_inputs(self.fixture(Path(temp)))
            self.assertEqual([r['frame_id'] for r in inputs['frames']], [f'f{i}' for i in range(52)])
            np.testing.assert_allclose(inputs['w2c'] @ inputs['c2w'], np.broadcast_to(np.eye(4), (52, 4, 4)), atol=1e-6)

    def test_accepts_actual_camera_pilot_world_without_gravity_or_yaw_alignment(self):
        with TemporaryDirectory() as temp:
            camera = self.fixture(Path(temp))
            path = camera / 'undistorted/cameras.json'
            data = json.loads(path.read_text())
            data['world'] = 'Relative, unscaled COLMAP world; no gravity or yaw alignment'
            path.write_text(json.dumps(data))
            self.assertEqual(len(pilot.prepare_inputs(camera)['frames']), 52)

    def test_rejects_failed_upstream_membership_hash_escape_and_improper_rotation(self):
        for fault in ['status', 'membership', 'hash', 'escape', 'rotation', 'world']:
            with self.subTest(fault=fault), TemporaryDirectory() as temp:
                camera = self.fixture(Path(temp))
                if fault == 'hash':
                    (camera / 'undistorted/images/f10.jpg').write_bytes(b'bad')
                else:
                    filename = 'status.json' if fault == 'status' else 'manifest.json' if fault == 'membership' else 'undistorted/cameras.json'
                    path = camera / filename
                    data = json.loads(path.read_text())
                    if fault == 'status': data['status'] = 'failed'
                    if fault == 'membership': data['selected_frames'] = data['selected_frames'][1:]
                    if fault == 'escape': data['image_root'] = '../..'
                    if fault == 'rotation': data['frames'][2]['c2w'][0][0] = -1
                    if fault == 'world': data['world'] = 'meters'
                    path.write_text(json.dumps(data))
                with self.assertRaises(ValueError): pilot.prepare_inputs(camera)

    def test_four_windows_cover_52_with_four_overlap_and_deterministic_selection(self):
        windows = pilot.plan_windows(52)
        self.assertEqual(windows, [list(range(s, s + 16)) for s in [0, 12, 24, 36]])
        chosen = pilot.select_window_frames(windows)
        self.assertEqual(len(chosen), 52)
        self.assertEqual(chosen[13]['window_index'], 0)
        self.assertEqual(chosen[14]['window_index'], 1)
        self.assertEqual(chosen[49]['local_index'], 13)
        self.assertEqual(len(pilot.select_window_frames(windows[:1])), 16)
        for a, b in zip(windows, windows[1:]): self.assertEqual(len(set(a) & set(b)), 4)
        with self.assertRaises(ValueError): pilot.plan_windows(51)

    def test_overlap_diagnostics_use_only_finite_positive_depth_and_are_not_fusion(self):
        a = np.array([[1, 2, np.nan], [0, 4, 8]], np.float32)
        b = np.array([[1, 2.1, 8], [3, 8, 8]], np.float32)
        result = pilot.depth_agreement(a, b)
        self.assertEqual(result['valid_pair_count'], 4)
        self.assertAlmostEqual(result['median_depth_ratio_b_over_a'], 1.025)
        self.assertEqual(result['agreement_within_10_percent'], 75)
        self.assertIsNone(pilot.depth_agreement(np.zeros((2, 2)), np.ones((2, 2)))['median_depth_ratio_b_over_a'])

    def test_sixteen_view_numeric_contract_rejects_camera_overwrite_mismatch(self):
        K = np.repeat(np.array([[[900., 0, 539], [0, 900, 963.5], [0, 0, 1]]]), 16, axis=0)
        E = np.repeat(np.eye(4)[None], 16, axis=0)
        from room_reconstruction.depth_da3 import expected_resize
        Ks, hw = expected_resize(K, [1078, 1927])
        p = SimpleNamespace(depth=np.ones((16, *hw), np.float32), conf=np.ones((16, *hw), np.float32),
            extrinsics=E[:, :3].copy(), intrinsics=Ks, processed_images=np.zeros((16, *hw, 3), np.uint8), is_metric=0)
        self.assertEqual(pilot.validate_prediction(p, K, E, [1078, 1927])['shape'], [16, 504, 280])
        p.extrinsics[15, 0, 3] = 1
        with self.assertRaises(ValueError): pilot.validate_prediction(p, K, E, [1078, 1927])

    def test_capture_records_scale_from_actual_ransac_call_and_preserves_raw_arrays(self):
        calls = []
        def align(raw, supplied, **kwargs):
            calls.append(kwargs)
            return np.eye(3), np.zeros(3), 2., supplied
        api = SimpleNamespace(align_poses_umeyama=align)
        class Model:
            def _align_to_input_extrinsics_intrinsics(self, ext, intr, p):
                _, _, scale, _ = api.align_poses_umeyama(p.extrinsics, ext.numpy(),
                    ransac=len(ext) >= 10, return_aligned=True, random_state=42)
                p.depth /= scale
                p.extrinsics = ext.numpy()[:, :3]
                p.intrinsics = intr.numpy()
                return p
        class Tensor:
            def __init__(self, a): self.a = a
            def numpy(self): return self.a
            def __len__(self): return len(self.a)
        model = Model()
        capture = pilot.attach_alignment_capture(model, api)
        p = SimpleNamespace(extrinsics=np.ones((16, 3, 4)), intrinsics=np.ones((16, 3, 3)), depth=np.full((16, 2, 2), 6.))
        model._align_to_input_extrinsics_intrinsics(Tensor(np.zeros((16, 4, 4))), Tensor(np.zeros((16, 3, 3))), p)
        self.assertEqual(calls, [dict(ransac=True, return_aligned=True, random_state=42)])
        self.assertEqual(capture['input_to_prediction_scale'], 2)
        self.assertTrue(np.all(capture['depth'] == 6))
        self.assertTrue(np.all(capture['extrinsics'] == 1))
        self.assertTrue(np.all(p.depth == 3))
        self.assertIs(api.align_poses_umeyama, align)

    def test_cli_failed_run_is_persisted_and_never_overwritten(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/run_depth_pilot.py'
        spec = importlib.util.spec_from_file_location('depth_pilot_cli_test', path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(cli, 'ROOT', root), patch.object(cli, 'run_pilot', side_effect=ValueError('fresh gate failed')):
                self.assertEqual(cli.main(['--run-id', 'pilot', '--max-windows', '1']), 2)
                self.assertEqual(cli.main(['--run-id', 'pilot']), 2)
            status = json.loads((root / 'outputs/pilot/status.json').read_text())
            self.assertEqual(status['status'], 'failed')
            self.assertTrue((root / 'outputs/pilot/report.json').is_file())
            self.assertTrue((root / 'outputs/pilot/manifest.json').is_file())

    def test_fresh_guard_precedes_numpy_torch_and_model_imports(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/run_depth_pilot.py'
        spec = importlib.util.spec_from_file_location('depth_pilot_order_test', path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        import builtins
        original = builtins.__import__
        imports = []
        def record(name, *args, **kwargs):
            imports.append(name)
            return original(name, *args, **kwargs)
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            with patch.object(cli, 'load_yaml', return_value=dict(profile='server', paths=dict(project_root=str(root)))), \
                    patch.object(cli.sys, 'prefix', str(root / '.venv-inference')), \
                    patch.object(cli, 'fresh_resource_guard', side_effect=ValueError('fresh gate failed')), \
                    patch.object(builtins, '__import__', side_effect=record):
                with self.assertRaisesRegex(ValueError, 'fresh gate failed'):
                    cli.run_pilot(root, root, SimpleNamespace(config='config.yaml'))
        self.assertFalse(any(name.startswith(('numpy', 'torch', 'depth_anything_3')) for name in imports))

    def test_smoke_reference_requires_matching_bytes_and_accepted_status(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/run_depth_pilot.py'
        spec = importlib.util.spec_from_file_location('depth_pilot_reference_test', path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / 'outputs/smoke'
            run.mkdir(parents=True)
            frames, artifacts = [], []
            for i in range(4):
                file = run / f'f{i}.npz'
                file.write_bytes(b'example artifact')
                frames.append(dict(frame_id=f'f{i}', image_sha256=str(i)))
                artifacts.append(dict(frame_id=f'f{i}', file=file.name, sha256=sha256_file(file)))
            for name, data in [('status', dict(status='smoke_passed')),
                    ('manifest', dict(camera_run='camera', selected_train_frames=frames)),
                    ('report', dict(status='smoke_passed', checkpoint=dict(weights_sha256='weights'),
                        vendor_source_hashes=dict(api='source'), artifacts=artifacts))]:
                (run / f'{name}.json').write_text(json.dumps(data))
            args = SimpleNamespace(smoke_run='smoke', camera_run='camera')
            reference, hashes = cli.smoke_reference(root, args, dict(frames=frames), dict(weights_sha256='weights'), dict(api='source'))
            self.assertEqual(len(reference), 4)
            self.assertEqual(len(hashes), 3)
            (run / 'f0.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                cli.smoke_reference(root, args, dict(frames=frames), dict(weights_sha256='weights'), dict(api='source'))


if __name__ == '__main__': unittest.main()
