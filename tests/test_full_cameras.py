import importlib.util
import builtins
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from room_reconstruction import full_cameras
from room_reconstruction.provenance import sha256_file


def frames(count=302):
    return [dict(frame_id=f'f{i:03}', timestamp_seconds=i / 2 + .0167,
                 source_index=i, split='validation' if i % 5 == 4 else 'train',
                 image_path=f'frames/f{i:03}.jpg', image_sha256=str(i), display_size=[1080, 1920])
            for i in range(count)]


class FullCameraTests(unittest.TestCase):
    def test_full_selection_keeps_all_302_and_original_audit_splits(self):
        original = frames()
        selected = full_cameras.select_full_frames(list(reversed(original)))
        self.assertEqual(len(selected), 302)
        self.assertEqual(sum(row['audit_split'] == 'validation' for row in selected), 60)
        self.assertEqual(selected[0]['timestamp_seconds'], .0167)
        self.assertEqual(selected[-1]['timestamp_seconds'], 150.5167)
        self.assertTrue(all(row['reconstruction_role'] == 'production' for row in selected))
        self.assertEqual(selected[4]['split'], 'validation')
        self.assertNotIn('audit_split', original[4])

    def test_full_selection_rejects_ambiguous_ids_names_and_invalid_time(self):
        for key, value in [('frame_id', 'f000'), ('image_path', 'other/f000.jpg'),
                           ('timestamp_seconds', float('nan')), ('split', 'test')]:
            with self.subTest(key=key):
                rows = frames(3)
                rows[1][key] = value
                with self.assertRaises(ValueError):
                    full_cameras.select_full_frames(rows)

    def test_small_component_passes_without_global_coverage_gate(self):
        cameras = {'frames': [dict(K=[[800, 0, 540], [0, 900, 960], [0, 0, 1]],
                                  camera_params=[800, 900, 540, 960, 0, 0, 0, 0],
                                  image_size=[1080, 1920], c2w=np.eye(4).tolist()) for _ in range(3)]}
        quality = full_cameras.component_quality([.1, .5, 1], cameras)
        self.assertTrue(quality['passed'])
        self.assertNotIn('registration_rate_min', quality['thresholds'])
        cameras['frames'][0]['K'][0][0] = -1
        self.assertFalse(full_cameras.component_quality([.1], cameras)['passed'])
        cameras['frames'][0]['K'][0][0] = 800
        cameras['frames'][0]['c2w'][0][0] = -1
        self.assertFalse(full_cameras.component_quality([.1], cameras)['passed'])
        self.assertFalse(full_cameras.component_quality([], cameras)['passed'])

    def test_coverage_accounts_for_overlap_rejected_models_and_gaps(self):
        rows = frames(7)
        components = [dict(component_id='component_000', accepted=True, frame_ids=['f000', 'f001', 'f002']),
                      dict(component_id='component_001', accepted=True, frame_ids=['f002', 'f003', 'f004']),
                      dict(component_id='component_002', accepted=False, frame_ids=['f005'])]
        result = full_cameras.coverage_summary(rows, components)
        self.assertEqual(result['registered_union_count'], 6)
        self.assertEqual(result['accepted_union_count'], 5)
        self.assertEqual(result['missing_frame_ids'], ['f005', 'f006'])
        self.assertEqual(result['unregistered_frame_ids'], ['f006'])
        self.assertFalse(result['coverage_complete'])
        self.assertTrue(result['partial_coverage'])
        self.assertEqual(len(result['frame_assignments']), 7)
        duplicate = result['frame_assignments'][2]
        self.assertEqual(duplicate['primary_component_id'], 'component_000')
        self.assertEqual(duplicate['component_ids'], ['component_000', 'component_001'])
        self.assertEqual(result['missing_frames'][0]['timestamp_seconds'], rows[5]['timestamp_seconds'])
        components.append(dict(component_id='component_003', accepted=True, frame_ids=['f005', 'f006']))
        self.assertTrue(full_cameras.coverage_summary(rows, components)['coverage_complete'])

    def test_empty_models_account_for_every_frame_as_unregistered(self):
        result = full_cameras.coverage_summary(frames(3), [])
        self.assertEqual(result['registered_union_count'], 0)
        self.assertEqual(result['accepted_union_count'], 0)
        self.assertEqual(result['missing_frame_ids'], ['f000', 'f001', 'f002'])
        self.assertTrue(result['all_selected_frames_accounted'])
        self.assertTrue(all(row['primary_component_id'] is None for row in result['frame_assignments']))
        with self.assertRaisesRegex(ValueError, 'outside selected audit'):
            full_cameras.coverage_summary(frames(3), [dict(component_id='c', accepted=True, frame_ids=['unknown'])])

    def test_real_audit_selection_has_entire_observed_span(self):
        root = Path(__file__).resolve().parents[1]
        path = root / 'outputs/l2bp3_audit_local_002/frame_manifest.json'
        if not path.exists():
            path = root / 'outputs/archive/experiments_2026-10-16/flat/l2bp3_audit_local_002/frame_manifest.json'
        if not path.exists():
            self.skipTest('Local audit unavailable')
        selected = full_cameras.select_full_frames(json.loads(path.read_text())['frames'])
        self.assertEqual(len(selected), 302)
        self.assertEqual(sum(row['audit_split'] == 'validation' for row in selected), 60)
        self.assertAlmostEqual(selected[-1]['timestamp_seconds'], 150.823333, places=4)

    def test_production_camera_export_preserves_audit_split_and_scope(self):
        rows = frames(5)
        camera = SimpleNamespace(width=1080, height=1920, model_name='OPENCV',
            params=np.array([800, 900, 540, 960, 0, 0, 0, 0]),
            calibration_matrix=lambda: np.array([[800, 0, 540], [0, 900, 960], [0, 0, 1]]))
        image = SimpleNamespace(name='f004.jpg', camera_id=1,
            cam_from_world=lambda: SimpleNamespace(matrix=lambda: np.column_stack([np.eye(3), [1, 2, 3]])))
        model = SimpleNamespace(reg_image_ids=lambda: [1], images={1: image}, cameras={1: camera})
        result = full_cameras.production_cameras(model, rows, 'component_002')
        self.assertEqual(result['coordinate_frame_id'], 'component_002')
        self.assertIn('component-local', result['world'])
        row = result['frames'][0]
        self.assertEqual(row['split'], 'train')
        self.assertEqual(row['audit_split'], 'validation')
        self.assertEqual(row['reconstruction_role'], 'production')


class FullCameraCliTests(unittest.TestCase):
    def cli(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/reconstruct_full_cameras.py'
        spec = importlib.util.spec_from_file_location('full_camera_cli', path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        return cli

    def setup_run(self, root):
        audit, run = root / 'outputs/audit', root / 'outputs/full'
        audit.mkdir(parents=True)
        run.mkdir()
        (audit / 'frames').mkdir()
        rows = frames(7)
        for row in rows:
            path = audit / row['image_path']
            path.write_bytes(row['frame_id'].encode())
            row['image_sha256'] = sha256_file(path)
        (audit / 'status.json').write_text('{"stage":"video_audit","status":"complete"}')
        (audit / 'frame_manifest.json').write_text(json.dumps({'source_sha256': 'video-hash', 'frames': rows}))
        (root / 'config.yaml').write_text(f'profile: server\npaths:\n  project_root: {root}\n')
        args = SimpleNamespace(config='config.yaml', threads=4, audit_run='audit', run_id='full',
                               max_image_size=960, max_features=4096, camera_model='OPENCV',
                               matching='exhaustive', overlap=16, loop_detection=False)
        return audit, run, args

    def test_resource_rejection_precedes_numpy_colmap_import(self):
        cli = self.cli()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            _, run, args = self.setup_run(root)
            original_import = builtins.__import__
            def guarded_import(name, *args, **kwargs):
                if name in ('numpy', 'pycolmap', 'room_reconstruction.full_cameras', 'room_reconstruction.poses'):
                    raise AssertionError('Heavy import reached before resource gate')
                return original_import(name, *args, **kwargs)
            with patch.object(cli, 'evaluate_resources', return_value={'resource_ready': False, 'errors': ['low memory']}), \
                    patch('builtins.__import__', side_effect=guarded_import):
                with self.assertRaisesRegex(ValueError, 'low memory'):
                    cli.run_full(root, run, args)
            self.assertFalse((run / 'manifest.json').exists())

    def test_incomplete_audit_rejected_before_copying_frames(self):
        cli = self.cli()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            audit, run, args = self.setup_run(root)
            (audit / 'status.json').write_text('{"stage":"video_audit","status":"failed"}')
            with patch.object(cli, 'evaluate_resources', return_value={'resource_ready': True}):
                with self.assertRaisesRegex(ValueError, 'completed video_audit'):
                    cli.run_full(root, run, args)
            self.assertFalse((run / 'images').exists())

    def test_main_reports_accepted_partial_coverage_without_calling_it_complete(self):
        cli = self.cli()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            report = dict(camera_ready=True, coverage_complete=False, partial_coverage=True,
                          accepted_union_count=5, selected_frame_count=302)
            with patch.object(cli, 'ROOT', root), patch.object(cli, 'run_full', return_value=report) as run:
                result = cli.main(['--audit-run', 'audit', '--run-id', 'new_full'])
            self.assertEqual(result, 0)
            args = run.call_args.args[2]
            self.assertEqual((args.threads, args.max_image_size, args.max_features, args.camera_model),
                             (4, 960, 4096, 'OPENCV'))
            self.assertEqual((args.matching, args.overlap), ('exhaustive', 16))
            status = json.loads((root / 'outputs/new_full/status.json').read_text())
            self.assertEqual(status['status'], 'camera_passed')
            self.assertTrue(status['camera_ready'])
            self.assertFalse(status['coverage_complete'])
            self.assertTrue(status['partial_coverage'])

    def test_cpu_models_preserved_separately_and_accepted_components_undistorted(self):
        self.check_cpu_models('exhaustive')

    def test_sequential_cpu_pairing_preserves_full_selection_and_arguments(self):
        self.check_cpu_models('sequential')

    def test_sequential_loop_detection_flag_reaches_pairing(self):
        self.check_cpu_models('sequential', loop_detection=True)

    def test_nonpositive_overlap_rejected_before_reconstruction(self):
        cli = self.cli()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with patch.object(cli, 'ROOT', root), patch.object(cli, 'run_full') as run:
                result = cli.main(['--audit-run', 'audit', '--run-id', 'new_full', '--overlap', '0'])
            self.assertEqual(result, 2)
            run.assert_not_called()
            status = json.loads((root / 'outputs/new_full/status.json').read_text())
            self.assertIn('overlap', status['reason'])

    def check_cpu_models(self, matching_mode, loop_detection=False):
        cli = self.cli()
        class Model:
            def __init__(self, ids, undistorted=False):
                self.ids = ids
                width, height, focal = (800, 1420, 600) if undistorted else (1080, 1920, 800)
                camera = SimpleNamespace(width=width, height=height,
                    model_name='PINHOLE' if undistorted else 'OPENCV',
                    params=np.array([focal, focal, width / 2, height / 2]),
                    calibration_matrix=lambda: np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]]))
                self.cameras = {1: camera}
                self.points3D = {1: None}
                self.images = {i: SimpleNamespace(name=f'f{i:03}.jpg', camera_id=1,
                    cam_from_world=lambda: SimpleNamespace(matrix=lambda: np.column_stack([np.eye(3), [1, 2, 3]]))) for i in ids}
            def reg_image_ids(self):
                return self.ids
            def write(self, path):
                (Path(path) / 'native.bin').write_bytes(b'model')
        models = {8: Model([2, 3, 4]), 2: Model([0, 1, 2]), 9: Model([5])}
        undistorted = {}
        calls = []
        matching_calls = []
        expected_vocab_tree = None
        def undistort(output, primary, images, **kwargs):
            output = Path(output)
            (output / 'images').mkdir(parents=True)
            index = int(output.parent.name.rsplit('_', 1)[1])
            ids = [0, 1, 2] if index == 0 else [2, 3, 4]
            for i in ids:
                (output / 'images' / f'f{i:03}.jpg').write_bytes(b'undistorted')
            undistorted[str(output / 'sparse')] = Model(ids, True)
            calls.append(output.parent.name)
        options = lambda: SimpleNamespace(sift=SimpleNamespace(), mapper=SimpleNamespace())
        fake = SimpleNamespace(__version__='4.2.1', FeatureExtractionOptions=options,
            FeatureMatchingOptions=options, ImageReaderOptions=options, IncrementalPipelineOptions=options,
            SequentialPairingOptions=options,
            BundleAdjustmentBackend=SimpleNamespace(CERES='ceres'), CameraMode=SimpleNamespace(SINGLE='single'),
            Device=SimpleNamespace(cpu='cpu'), extract_features=lambda *a, **k: None,
            match_exhaustive=lambda *a, **k: None, incremental_mapping=lambda *a, **k: models,
            undistort_images=undistort, Reconstruction=lambda path: undistorted[str(path)])
        def extract(*args, **kwargs):
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(os.environ['OMP_NUM_THREADS'], '4')
            self.assertFalse(kwargs['extraction_options'].use_gpu)
            self.assertEqual(kwargs['device'], 'cpu')
            expected = (1600, 8192) if matching_mode == 'sequential' else (960, 4096)
            self.assertEqual((kwargs['extraction_options'].max_image_size,
                              kwargs['extraction_options'].sift.max_num_features), expected)
        def match(*args, **kwargs):
            matching_calls.append('exhaustive')
            self.assertFalse(kwargs['matching_options'].use_gpu)
            self.assertEqual(kwargs['device'], 'cpu')
        def sequential(*args, **kwargs):
            matching_calls.append('sequential')
            self.assertFalse(kwargs['matching_options'].use_gpu)
            self.assertEqual(kwargs['device'], 'cpu')
            pairing = kwargs['pairing_options']
            self.assertEqual(pairing.overlap, 16)
            self.assertEqual(pairing.num_threads, 4)
            self.assertFalse(pairing.quadratic_overlap)
            self.assertEqual(pairing.loop_detection, loop_detection)
            if loop_detection:
                self.assertEqual(pairing.vocab_tree_path, expected_vocab_tree)
        def mapping(*args, **kwargs):
            self.assertFalse(kwargs['options'].ba_use_gpu)
            self.assertEqual(kwargs['options'].ba_global_backend, 'ceres')
            return models
        fake.extract_features, fake.match_exhaustive, fake.incremental_mapping = extract, match, mapping
        fake.match_sequential = sequential
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            audit, run, args = self.setup_run(root)
            args.matching = matching_mode
            args.loop_detection = loop_detection
            args.vocab_tree_path = None
            if loop_detection:
                args.vocab_tree_path = str(root / 'vocab_tree.bin')
                Path(args.vocab_tree_path).write_bytes(b'vocab-tree-fixture')
                expected_vocab_tree = str(Path(args.vocab_tree_path).resolve())
            if matching_mode == 'sequential':
                args.max_image_size, args.max_features = 1600, 8192
            original_manifest = (audit / 'frame_manifest.json').read_bytes()
            with patch.object(cli, 'evaluate_resources', return_value={'resource_ready': True}), \
                    patch.dict('sys.modules', {'pycolmap': fake}), \
                    patch.dict(os.environ), \
                    patch('room_reconstruction.poses.observation_residuals', return_value=[.1, .5]):
                report = cli.run_full(root, run, args)
            self.assertEqual(matching_calls, [matching_mode])
            parent_manifest = json.loads((run / 'manifest.json').read_text())
            self.assertEqual(parent_manifest['arguments']['matching'], matching_mode)
            self.assertEqual(parent_manifest['arguments']['overlap'], 16)
            self.assertEqual(len(parent_manifest['selected_frames']), 7)
            self.assertEqual(report['matching_policy']['mode'], matching_mode)
            self.assertEqual(report['matching_policy']['device'], 'cpu')
            self.assertEqual(calls, ['component_000', 'component_001'])
            self.assertEqual(report['accepted_union_count'], 5)
            self.assertEqual(report['registered_union_count'], 6)
            self.assertEqual(report['missing_frame_ids'], ['f005', 'f006'])
            self.assertFalse(report['coverage_complete'])
            self.assertEqual([c['model_id'] for c in report['components']], [2, 8, 9])
            for index in range(3):
                component = run / 'components' / f'component_{index:03}'
                self.assertTrue((component / 'primary/native.bin').exists())
                self.assertEqual(json.loads((component / 'status.json').read_text())['camera_ready'], index < 2)
                manifest = json.loads((component / 'manifest.json').read_text())
                self.assertEqual(manifest['stage'], 'camera_full_component')
                if index == 1:
                    self.assertEqual(manifest['selected_frames'][-1]['split'], 'validation')
            pinhole = json.loads((run / 'components/component_001/undistorted/cameras.json').read_text())['frames'][-1]
            self.assertEqual(pinhole['audit_split'], 'validation')
            self.assertEqual(pinhole['image_size'], [800, 1420])
            self.assertEqual(pinhole['K'][0][0], 600)
            self.assertEqual((audit / 'frame_manifest.json').read_bytes(), original_manifest)


if __name__ == '__main__':
    unittest.main()
