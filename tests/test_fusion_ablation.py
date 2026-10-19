import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


def load_cli(name='run_fusion_ablation'):
    path = Path(__file__).resolve().parents[1] / f'scripts/{name}.py'
    spec = importlib.util.spec_from_file_location(f'{name}_test', path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


class AblationTests(unittest.TestCase):
    def fixture(self, root):
        cli = load_cli()
        rows = [dict(frame_id=f'f{i}', split='train', c2w=np.eye(4).tolist()) for i in range(52)]
        inputs = {}
        for name, q in [('q10', .1), ('q0', 0.)]:
            folder = root / 'outputs' / name
            folder.mkdir(parents=True)
            manifest = dict(stage='filtered_depth', camera_run='camera', depth_run='depth', mask_run='masks',
                selected_train_frames=rows, settings=dict(confidence_quantile=q, edge_relative=.08))
            (folder / 'manifest.json').write_text(json.dumps(manifest))
            inputs[name] = dict(selected_train_frames=rows, camera_run='camera', frames=[dict(frame_id=r['frame_id']) for r in rows],
                upstream_hashes={str(folder / 'manifest.json'): cli.fusion.sha256_file(folder / 'manifest.json')},
                filter_report=dict(depth_run='depth', mask_run='masks'))
        reference = root / 'outputs/reference'
        reference.mkdir()
        parameters = dict(voxel_length=.01597707430521647, sdf_trunc=4*.01597707430521647,
            depth_trunc=8.896440752506248, units=cli.fusion.UNITS)
        (reference / 'filtered.ply').write_bytes(b'accepted colored mesh')
        artifact = dict(file='filtered.ply', sha256=cli.fusion.sha256_file(reference / 'filtered.ply'))
        for name, value in dict(status=dict(stage='tsdf_mesh', status='mesh_passed'),
                manifest=dict(stage='tsdf_mesh', filter_run='q10', camera_run='camera', selected_train_frames=rows,
                    parameters=parameters, units=cli.fusion.UNITS, input_and_source_hashes=inputs['q10']['upstream_hashes']),
                report=dict(stage='tsdf_mesh', frame_count=52, units=cli.fusion.UNITS, parameters=parameters,
                    meshes=dict(filtered=dict(surface_nonempty=True, artifacts=[artifact])))).items():
            (reference / f'{name}.json').write_text(json.dumps(value))
        args = SimpleNamespace(filter_run='q10', no_confidence_filter_run='q0', reference_run='reference')
        return cli, inputs, args

    def test_fixed_parameters_and_confidence_choice(self):
        with TemporaryDirectory() as temp:
            root = Path(temp); cli, inputs, args = self.fixture(root)
            with patch.object(cli.fusion, 'prepare_filter', side_effect=lambda root, name: inputs[name]):
                prepared = cli.prepare_ablation(root, args)
            a = cli.variant_parameters(prepared['parameters'], 'voxel2_trunc4')
            b = cli.variant_parameters(prepared['parameters'], 'confidence0_voxel2_trunc4')
            self.assertEqual(a, b)
            self.assertEqual(a['depth_trunc'], 8.896440752506248)
            self.assertAlmostEqual(a['voxel_length'], .01597707430521647 * 2)
            self.assertAlmostEqual(a['sdf_trunc'], a['voxel_length'] * 4)
            self.assertEqual(cli.VARIANTS['confidence0_voxel2_trunc4'][2], 0.)

    def test_rejects_uncontrolled_pair_and_incomplete_reference(self):
        for fault in ('quantile', 'depth', 'mask', 'camera', 'order', 'settings', 'partial', 'meshhash', 'parameters', 'reference_inputs'):
            with self.subTest(fault=fault), TemporaryDirectory() as temp:
                root = Path(temp); cli, inputs, args = self.fixture(root)
                path = root / 'outputs/q0/manifest.json'
                data = json.loads(path.read_text())
                if fault == 'quantile': data['settings']['confidence_quantile'] = .1
                if fault in ('depth', 'mask', 'camera'): data[f'{fault}_run'] = 'different'
                if fault == 'order': inputs['q0']['frames'].reverse()
                if fault == 'settings': data['settings']['edge_relative'] = .2
                path.write_text(json.dumps(data))
                if fault == 'partial':
                    (root / 'outputs/reference/status.json').write_text(json.dumps(dict(stage='tsdf_mesh', status='failed')))
                if fault == 'meshhash': (root / 'outputs/reference/filtered.ply').write_bytes(b'changed')
                if fault == 'parameters':
                    path = root / 'outputs/reference/report.json'; data = json.loads(path.read_text())
                    data['parameters']['depth_trunc'] = 1.; path.write_text(json.dumps(data))
                if fault == 'reference_inputs':
                    path = root / 'outputs/reference/manifest.json'; data = json.loads(path.read_text())
                    data['input_and_source_hashes'] = {}; path.write_text(json.dumps(data))
                with patch.object(cli.fusion, 'prepare_filter', side_effect=lambda root, name: inputs[name]):
                    with self.assertRaises(ValueError): cli.prepare_ablation(root, args)

    def test_boundary_plane_closed_tetrahedron_and_nonmanifold_edge(self):
        cli = load_cli()
        vertices = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]], float)
        result = cli.boundary_diagnostics(vertices, np.array([[0,1,2], [0,2,3]]))
        self.assertEqual(result['boundary_edge_count'], 4)
        self.assertEqual(result['closed_boundary_loop_count'], 1)
        self.assertEqual(result['branched_boundary_component_count'], 0)
        self.assertAlmostEqual(result['boundary_length'], 4.)
        self.assertAlmostEqual(result['surface_area'], 1.)
        self.assertAlmostEqual(result['boundary_length_over_sqrt_area'], 4.)
        closed = cli.boundary_diagnostics(vertices, np.array([[0,1,2], [0,3,1], [1,3,2], [0,2,3]]))
        self.assertEqual(closed['boundary_edge_count'], 0)
        self.assertEqual(closed['closed_boundary_loop_count'], 0)
        vertices = np.vstack([vertices, [0,0,1]])
        branched = cli.boundary_diagnostics(vertices, np.array([[0,1,2], [1,0,3], [0,1,4]]))
        self.assertEqual(branched['boundary_edge_count'], 6)
        self.assertEqual(branched['nonmanifold_edge_count'], 1)
        self.assertEqual(branched['branched_boundary_component_count'], 1)

    def test_failed_variant_and_collision_never_claim_full_suite(self):
        cli = load_cli()
        argv = ['--filter-run', 'q10', '--no-confidence-filter-run', 'q0', '--reference-run', 'reference',
            '--run-id', 'experiment', '--variant', 'voxel1p5_trunc4']
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(cli, 'ROOT', root), patch.object(cli, 'execute', side_effect=ValueError('resource gate failed')):
                self.assertEqual(cli.main(argv), 2)
                before = (root / 'outputs/experiment/status.json').read_bytes()
                self.assertEqual(cli.main(argv), 2)
            report = json.loads((root / 'outputs/experiment/report.json').read_text())
            self.assertEqual(report['requested_variants'], ['voxel1p5_trunc4'])
            self.assertEqual(report['completed_variants'], [])
            self.assertEqual((root / 'outputs/experiment/status.json').read_bytes(), before)

    def test_filter_quantile_cli_validation_and_default(self):
        cli = load_cli('filter_depth')
        argv = ['--depth-run', 'depth', '--mask-run', 'masks', '--run-id', 'new']
        for q in ('-0.1', '1', 'nan', 'inf'):
            with self.subTest(q=q), patch.object(cli, 'create_run_dir') as create:
                with self.assertRaises(SystemExit): cli.main(argv + ['--confidence-quantile', q])
                create.assert_not_called()
        for value, extra in ((.1, []), (0., ['--confidence-quantile', '0'])):
            with TemporaryDirectory() as temp, patch.object(cli, 'create_run_dir', return_value=Path(temp)), \
                    patch.object(cli, 'execute', return_value=dict(mean_base_coverage=1., mean_filtered_coverage=1.)) as execute:
                self.assertEqual(cli.main(argv + extra), 0)
                self.assertEqual(execute.call_args.args[1].confidence_quantile, value)

    def test_cpu_gate_precedes_numeric_backend_imports(self):
        cli = load_cli()
        import builtins
        original = builtins.__import__
        imports = []
        def record(name, *args, **kwargs):
            imports.append(name)
            return original(name, *args, **kwargs)
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            with patch.object(cli.fusion, 'load_yaml', return_value=dict(profile='server', paths=dict(project_root=str(root)))), \
                    patch.object(cli.sys, 'prefix', str(root / '.venv-geometry')), \
                    patch.object(cli.fusion, 'cpu_guard', side_effect=ValueError('fresh CPU gate failed')), \
                    patch.object(builtins, '__import__', side_effect=record):
                with self.assertRaisesRegex(ValueError, 'fresh CPU gate failed'):
                    cli.execute(root, root, SimpleNamespace(config='config.yaml'))
        self.assertFalse(any(name.startswith(('numpy', 'open3d')) for name in imports))

    def test_completed_q0_report_contains_only_requested_variant(self):
        cli = load_cli()
        argv = ['--no-confidence-filter-run', 'q0', '--run-id', 'experiment',
            '--variant', 'confidence0_voxel2_trunc4']
        report = dict(stage=cli.STAGE, status='ablation_passed',
            requested_variants=['confidence0_voxel2_trunc4'], completed_variants=['confidence0_voxel2_trunc4'])
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(cli, 'ROOT', root), patch.object(cli, 'execute', return_value=report):
                self.assertEqual(cli.main(argv), 0)
            status = json.loads((root / 'outputs/experiment/status.json').read_text())
            self.assertEqual(status['completed_variants'], ['confidence0_voxel2_trunc4'])
            self.assertEqual(json.loads((root / 'outputs/experiment/report.json').read_text()), report)


if __name__ == '__main__': unittest.main()
