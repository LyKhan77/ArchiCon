import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory

import numpy as np

from room_reconstruction import depth_da3 as da3
from room_reconstruction.provenance import sha256_file


class DA3ContractTests(unittest.TestCase):
    def fixture(self, root):
        camera = root / 'camera'
        (camera / 'undistorted' / 'images').mkdir(parents=True)
        frames = []
        for i in range(14):
            image = camera / 'undistorted' / 'images' / f'frame_{i}.jpg'
            image.write_bytes(f'image {i}'.encode())
            pose = np.eye(4)
            pose[:3, 3] = [i, i * i / 20, 0]
            frames.append(dict(frame_id=f'f{i}', image_name=image.name,
                image_sha256=sha256_file(image), timestamp_seconds=19 + i / 2,
                source_index=i, split='validation' if i == 4 else 'train',
                image_size=[1078, 1927], camera_model='PINHOLE',
                K=[[900, 0, 539], [0, 900, 963.5], [0, 0, 1]], c2w=pose.tolist()))
        for name, data in [('status', {'status': 'camera_passed', 'camera_ready': True}),
                           ('report', {'stage': 'camera_pilot', 'quality': {'passed': True}, 'undistorted': True}),
                           ('manifest', {'stage': 'camera_pilot', 'selected_frames': [f for f in frames if f['split'] == 'train']})]:
            (camera / f'{name}.json').write_text(json.dumps(data))
        (camera / 'undistorted' / 'cameras.json').write_text(json.dumps({
            'schema_version': 1, 'pose_convention': 'OpenCV c2w; +x right, +y down, +z forward',
            'world': 'Relative, unscaled COLMAP world', 'image_root': 'images', 'frames': frames}))
        return camera

    def test_four_train_views_metadata_relative_paths_and_inverse_pose(self):
        with TemporaryDirectory() as temp:
            camera = self.fixture(Path(temp))
            result = da3.prepare_inputs(camera)
            rows = result['frames']
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row['split'] == 'train' and 20 <= row['timestamp_seconds'] <= 26 for row in rows))
            self.assertEqual([row['timestamp_seconds'] for row in rows], [20, 22, 24, 25.5])
            self.assertTrue(all(Path(path).parent == (camera / 'undistorted' / 'images').resolve() for path in result['image_paths']))
            np.testing.assert_allclose(result['w2c'] @ result['c2w'], np.broadcast_to(np.eye(4), (4, 4, 4)), atol=1e-6)

    def test_rejects_failed_upstream_hash_change_and_image_escape(self):
        for fault in ['status', 'hash', 'escape', 'train_membership']:
            with self.subTest(fault=fault), TemporaryDirectory() as temp:
                camera = self.fixture(Path(temp))
                if fault == 'status':
                    (camera / 'status.json').write_text('{"status":"failed","camera_ready":true}')
                elif fault == 'hash':
                    (camera / 'undistorted/images/frame_2.jpg').write_bytes(b'tampered')
                elif fault == 'escape':
                    p = camera / 'undistorted/cameras.json'
                    data = json.loads(p.read_text())
                    data['image_root'] = '../..'
                    p.write_text(json.dumps(data))
                else:
                    p = camera / 'manifest.json'
                    data = json.loads(p.read_text())
                    data['selected_frames'] = [r for r in data['selected_frames'] if r['frame_id'] != 'f2']
                    p.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    da3.prepare_inputs(camera)

    def test_rejects_invalid_camera_and_held_out_shortage(self):
        for fault in ['rotation', 'split', 'intrinsics']:
            with self.subTest(fault=fault), TemporaryDirectory() as temp:
                camera = self.fixture(Path(temp))
                p = camera / 'undistorted/cameras.json'
                data = json.loads(p.read_text())
                for row in data['frames']:
                    if fault == 'rotation': row['c2w'][0][0] = 2
                    if fault == 'split': row['split'] = 'validation'
                    if fault == 'intrinsics': row['K'][0][0] = -1
                p.write_text(json.dumps(data))
                with self.assertRaises(ValueError): da3.prepare_inputs(camera)

    def test_output_contract_accepts_some_invalid_depth_but_reports_fraction(self):
        K = np.repeat(np.array([[[900, 0, 539], [0, 900, 963.5], [0, 0, 1]]]), 4, axis=0)
        Ks, hw = da3.expected_resize(K, [1078, 1927])
        self.assertEqual(hw, (504, 280))
        np.testing.assert_allclose(Ks[0], [[233.76623377, 0, 140], [0, 235.39180073, 252], [0, 0, 1]])
        E = np.repeat(np.eye(4)[None], 4, axis=0)
        prediction = SimpleNamespace(depth=np.ones((4, *hw), np.float32),
            conf=np.ones((4, *hw), np.float32), intrinsics=Ks,
            extrinsics=E[:, :3].copy(), processed_images=np.zeros((4, *hw, 3), np.uint8), is_metric=0)
        prediction.depth[0, 0, 0] = np.nan
        summary = da3.validate_prediction(prediction, K, E, [1078, 1927])
        self.assertLess(summary['frames'][0]['valid_depth_fraction'], 1)
        for field in ['intrinsics', 'extrinsics', 'conf', 'depth']:
            with self.subTest(field=field):
                original = getattr(prediction, field).copy()
                if field == 'intrinsics': prediction.intrinsics[0, 0, 0] += 1
                elif field == 'extrinsics': prediction.extrinsics[0, 0, 3] += 1
                elif field == 'conf': prediction.conf[0, 0, 0] = np.inf
                else: prediction.depth[:] = 0
                with self.assertRaises(ValueError): da3.validate_prediction(prediction, K, E, [1078, 1927])
                setattr(prediction, field, original)

    def test_model_snapshot_rejects_wrong_model_config_and_weight_size(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            entry = dict(model_id='depth-anything/DA3-LARGE', revision='a' * 40,
                cache_directory='models--depth-anything--DA3-LARGE', weights_bytes=10)
            snapshot = root / entry['cache_directory'] / 'snapshots' / entry['revision']
            snapshot.mkdir(parents=True)
            (snapshot / 'model.safetensors').write_bytes(b'1234567890')
            (snapshot / 'config.json').write_text('{"model_name":"da3-large"}')
            self.assertEqual(da3.validate_snapshot(root, entry)['snapshot'], str(snapshot))
            (snapshot / 'model.safetensors').write_bytes(b'bad')
            with self.assertRaises(ValueError): da3.validate_snapshot(root, entry)
            (snapshot / 'model.safetensors').write_bytes(b'1234567890')
            (snapshot / 'config.json').write_text('{"model_name":"da3metric-large"}')
            with self.assertRaises(ValueError): da3.validate_snapshot(root, entry)

    def test_cli_persists_failed_status_manifest_and_report(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/smoke_da3.py'
        spec = importlib.util.spec_from_file_location('smoke_da3_test', path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(cli, 'ROOT', root), patch.object(cli, 'run_smoke', side_effect=ValueError('resource gate rejected')):
                self.assertEqual(cli.main(['--camera-run', 'camera', '--run-id', 'smoke']), 2)
            status = json.loads((root / 'outputs/smoke/status.json').read_text())
            self.assertEqual(status['status'], 'failed')
            self.assertFalse(status['inference_ready'])
            self.assertTrue((root / 'outputs/smoke/manifest.json').is_file())
            self.assertTrue((root / 'outputs/smoke/report.json').is_file())

    def test_run_guard_rejects_before_numpy_torch_or_da3_import(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/smoke_da3.py'
        spec = importlib.util.spec_from_file_location('smoke_da3_order_test', path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        import builtins
        original_import = builtins.__import__
        imports = []
        def record_import(name, *args, **kwargs):
            imports.append(name)
            return original_import(name, *args, **kwargs)
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            config = {'profile': 'server', 'paths': {'project_root': str(root)}}
            with patch.object(cli, 'load_yaml', return_value=config), \
                    patch.object(cli.sys, 'prefix', str(root / '.venv-inference')), \
                    patch.object(cli, 'fresh_resource_guard', side_effect=ValueError('fresh gate failed')), \
                    patch.object(builtins, '__import__', side_effect=record_import):
                with self.assertRaisesRegex(ValueError, 'fresh gate failed'):
                    cli.run_smoke(root, root, SimpleNamespace(config='config.yaml'))
        self.assertFalse(any(name.startswith(('numpy', 'torch', 'depth_anything_3')) for name in imports))

    def test_fresh_guard_rejects_busy_selected_gpu_without_changing_visibility(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/smoke_da3.py'
        spec = importlib.util.spec_from_file_location('smoke_da3_guard_test', path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        uuid = 'GPU-00000000-0000-4000-8000-000000000001'
        replies = [SimpleNamespace(stdout=f'0, {uuid}, Test GPU, 32768, 32000, 0\n'), SimpleNamespace(stdout=uuid)]
        with TemporaryDirectory() as temp, patch.object(cli.subprocess, 'run', side_effect=replies), \
                patch.object(cli.Path, 'read_text', return_value='MemAvailable: 16777216 kB\n'), \
                patch.object(cli.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 * 1024**3)), \
                patch.dict(cli.os.environ, {'CUDA_VISIBLE_DEVICES': uuid}):
            with self.assertRaisesRegex(ValueError, 'active compute processes'):
                cli.fresh_resource_guard(Path(temp), {'compute': {'gpu_selector': uuid}}, 'resources.json')
            self.assertEqual(cli.os.environ['CUDA_VISIBLE_DEVICES'], uuid)


if __name__ == '__main__':
    unittest.main()
