import importlib.util
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from room_reconstruction.provenance import sha256_file
from room_reconstruction.depth_da3 import expected_resize


def load_cli():
    path = Path(__file__).resolve().parents[1] / 'scripts/fuse_tsdf.py'
    spec = importlib.util.spec_from_file_location('fuse_tsdf_cli_test', path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


class FuseCLITests(unittest.TestCase):
    def fixture(self, root):
        run = root / 'outputs/filtered'
        run.mkdir(parents=True)
        native_K = [[900., 0, 539], [0, 900, 963.5], [0, 0, 1]]
        rows = [dict(frame_id=f'f{i}', split='train', K=native_K, image_size=[1078, 1927],
            c2w=np.eye(4).tolist()) for i in range(52)]
        K, shape = expected_resize(np.array([native_K]), [1078, 1927])
        stream = BytesIO()
        np.savez_compressed(stream, depth=np.full(shape, 3., np.float32), conf=np.ones(shape, np.float32),
            processed_rgb=np.full((*shape, 3), 128, np.uint8), K=K[0], w2c=np.eye(4)[:3],
            base_mask=np.ones(shape, bool), mask=np.ones(shape, bool),
            normal=np.zeros((*shape, 3), np.float32), normal_valid=np.zeros(shape, bool))
        artifacts = []
        for row in rows:
            path = run / f'{row["frame_id"]}.npz'
            path.write_bytes(stream.getvalue())
            artifacts.append(dict(frame_id=row['frame_id'], file=path.name, sha256=sha256_file(path)))
        values = dict(status=dict(stage='filtered_depth', status='filter_passed'),
            manifest=dict(stage='filtered_depth', camera_run='l2bp3_camera_pilot_002', selected_train_frames=rows),
            report=dict(stage='filtered_depth', status='filter_passed', camera_run='l2bp3_camera_pilot_002',
                frame_count=52, frames=artifacts, units='relative COLMAP world units; not meters'))
        for name, value in values.items():
            (run / f'{name}.json').write_text(json.dumps(value))
        return run, rows

    def test_rejects_partial_stage_and_non_train_provenance(self):
        cli = load_cli()
        for fault in ('partial', 'membership', 'hash', 'escape', 'duplicate', 'units'):
            with self.subTest(fault=fault), TemporaryDirectory() as temp:
                root = Path(temp)
                run, rows = self.fixture(root)
                name = 'status' if fault == 'partial' else 'manifest' if fault == 'membership' else 'report'
                path = run / f'{name}.json'
                data = json.loads(path.read_text())
                if fault == 'partial': data['status'] = 'resource_proof_passed'
                if fault == 'membership': data['selected_train_frames'][0]['split'] = 'validation'
                if fault == 'hash': data['frames'][0]['sha256'] = '0' * 64
                if fault == 'escape': data['frames'][0]['file'] = '../f0.npz'
                if fault == 'duplicate': data['frames'][1]['frame_id'] = 'f0'
                if fault == 'units': data['units'] = 'meters'
                path.write_text(json.dumps(data))
                with patch.object(cli, 'camera_inputs', return_value=dict(frames=rows, upstream_hashes={})):
                    with self.assertRaises(ValueError): cli.prepare_filter(root, 'filtered')

    def test_streams_validated_frames_and_derives_shared_scene_parameters(self):
        cli = load_cli()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _, rows = self.fixture(root)
            with patch.object(cli, 'camera_inputs', return_value=dict(frames=rows, upstream_hashes={})):
                inputs = cli.prepare_filter(root, 'filtered')
            params = cli.scene_parameters(inputs)
            self.assertAlmostEqual(params['voxel_length'], 3 / 150)
            self.assertAlmostEqual(params['sdf_trunc'], 4 * 3 / 150)
            self.assertAlmostEqual(params['depth_trunc'], 3.3)
            frames = cli.iter_frames(inputs, 'mask')
            self.assertIs(iter(frames), frames)
            frame, mask = next(frames)
            self.assertEqual(frame['depth'].dtype, np.float32)
            self.assertEqual(int(mask.sum()), 504 * 280)
            path = inputs['paths'][1]
            with np.load(path, allow_pickle=False) as archive:
                values = dict(archive)
            values['mask'] = values['mask'].astype(np.uint8)
            np.savez_compressed(path, **values)
            with self.assertRaisesRegex(ValueError, 'mask'): next(frames)

    def test_rejects_proper_but_wrong_frame_pose_and_intrinsics(self):
        cli = load_cli()
        for fault in ('pose', 'intrinsics'):
            with self.subTest(fault=fault), TemporaryDirectory() as temp:
                root = Path(temp)
                _, rows = self.fixture(root)
                with patch.object(cli, 'camera_inputs', return_value=dict(frames=rows, upstream_hashes={})):
                    inputs = cli.prepare_filter(root, 'filtered')
                path = inputs['paths'][0]
                with np.load(path, allow_pickle=False) as archive:
                    values = dict(archive)
                if fault == 'pose': values['w2c'][0, 3] = 1.
                if fault == 'intrinsics': values['K'][0, 0] *= 1.1
                np.savez_compressed(path, **values)
                with self.assertRaisesRegex(ValueError, 'camera'):
                    next(cli.iter_frames(inputs, 'mask'))

    def test_failed_run_persists_status_and_collision_never_overwrites(self):
        cli = load_cli()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(cli, 'ROOT', root), patch.object(cli, 'run_fusion', side_effect=ValueError('fresh CPU gate failed')):
                self.assertEqual(cli.main(['--filter-run', 'filtered', '--run-id', 'mesh']), 2)
                before = (root / 'outputs/mesh/status.json').read_bytes()
                self.assertEqual(cli.main(['--filter-run', 'filtered', '--run-id', 'mesh']), 2)
            self.assertEqual((root / 'outputs/mesh/status.json').read_bytes(), before)
            self.assertEqual(json.loads(before)['status'], 'failed')
            for name in ('manifest', 'report'): self.assertTrue((root / f'outputs/mesh/{name}.json').is_file())

    def test_cpu_gate_precedes_numpy_and_open3d_imports(self):
        cli = load_cli()
        import builtins
        original = builtins.__import__
        imports = []
        def record(name, *args, **kwargs):
            imports.append(name)
            return original(name, *args, **kwargs)
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            with patch.object(cli, 'load_yaml', return_value=dict(profile='server', paths=dict(project_root=str(root)))), \
                    patch.object(cli.sys, 'prefix', str(root / '.venv-geometry')), \
                    patch.object(cli, 'cpu_guard', side_effect=ValueError('fresh CPU gate failed')), \
                    patch.object(builtins, '__import__', side_effect=record):
                with self.assertRaisesRegex(ValueError, 'fresh CPU gate failed'):
                    cli.run_fusion(root, root, type('Args', (), dict(config='config.yaml'))())
        self.assertFalse(any(name.startswith(('numpy', 'open3d')) for name in imports))

    def test_rejects_regressed_open3d_020_before_processing_inputs(self):
        cli = load_cli()
        import builtins
        original = builtins.__import__
        def import_stale(name, *args, **kwargs):
            return SimpleNamespace(__version__='0.20.0') if name == 'open3d' else original(name, *args, **kwargs)
        for installed_version in ('0.20.0', '0.19.0'):
          with self.subTest(installed_version=installed_version), TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            with patch.object(cli, 'load_yaml', return_value=dict(profile='server', paths=dict(project_root=str(root)))), \
                    patch.object(cli.sys, 'prefix', str(root / '.venv-geometry')), \
                    patch.object(cli, 'cpu_guard', return_value={}), patch.object(np, '__version__', '1.26.4'), \
                    patch.object(cli.importlib.metadata, 'version', return_value=installed_version), \
                    patch.object(builtins, '__import__', side_effect=import_stale), \
                    patch.object(cli, 'prepare_filter') as prepare:
                with self.assertRaisesRegex(ValueError, 'Open3D 0.19.0; found 0.20.0'):
                    cli.run_fusion(root, root, type('Args', (), dict(config='config.yaml'))())
                prepare.assert_not_called()


if __name__ == '__main__': unittest.main()
