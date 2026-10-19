#!/usr/bin/env python3
"""One controlled cached-depth CPU TSDF variant per fresh process; no DA3 inference."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import fuse_tsdf as fusion

STAGE = 'fusion_ablation'
VARIANTS = {
    'voxel1p5_trunc4': (1.5, 4, .1),
    'voxel2_trunc4': (2., 4, .1),
    'voxel1p5_trunc8': (1.5, 8, .1),
    'voxel2_trunc8': (2., 8, .1),
    'confidence0_voxel2_trunc4': (2., 4, 0.),
}


def prepare_ablation(root, args):
    """Require accepted stages with identical cached inputs, cameras, order and settings."""
    if args.filter_run == args.no_confidence_filter_run:
        raise ValueError('Confidence comparison requires separate q10 and q0 filter runs')
    inputs = {name: fusion.prepare_filter(root, name)
        for name in (args.filter_run, args.no_confidence_filter_run)}
    manifests = []
    hashes = {}
    for name, q in ((args.filter_run, .1), (args.no_confidence_filter_run, 0.)):
        path = fusion.run_path(root, name) / 'manifest.json'
        manifest = json.loads(path.read_text())
        if manifest.get('settings', {}).get('confidence_quantile') != q:
            raise ValueError(f'{name} must declare confidence_quantile={q}')
        if any(manifest.get(key) != inputs[name]['filter_report'].get(key)
                for key in ('depth_run', 'mask_run')) or manifest.get('camera_run') != inputs[name]['camera_run']:
            raise ValueError('Filter manifest/report source mismatch')
        manifests.append(manifest)
        hashes.update(inputs[name]['upstream_hashes'])
        hashes[str(path)] = fusion.sha256_file(path)
    first, second = manifests
    if (any(not first.get(key) or first.get(key) != second.get(key)
            for key in ('depth_run', 'mask_run', 'camera_run', 'selected_train_frames'))
            or {k: v for k, v in first['settings'].items() if k != 'confidence_quantile'} !=
               {k: v for k, v in second['settings'].items() if k != 'confidence_quantile'}
            or inputs[args.filter_run]['selected_train_frames'] != inputs[args.no_confidence_filter_run]['selected_train_frames']
            or [r['frame_id'] for r in inputs[args.filter_run]['frames']] !=
               [r['frame_id'] for r in inputs[args.no_confidence_filter_run]['frames']]):
        raise ValueError('q10/q0 filters must share depth, masks, global cameras, frame order and all other settings')
    reference = fusion.run_path(root, args.reference_run)
    paths = [reference / f'{name}.json' for name in ('status', 'manifest', 'report')]
    if any(not path.resolve().is_relative_to(reference) for path in paths):
        raise ValueError('Reference metadata escapes run')
    state, manifest, report = [json.loads(path.read_text()) for path in paths]
    parameters = manifest.get('parameters', {})
    if (state.get('stage') != fusion.STAGE or state.get('status') != 'mesh_passed'
            or manifest.get('stage') != fusion.STAGE or report.get('stage') != fusion.STAGE
            or report.get('frame_count') != 52 or report.get('units') != fusion.UNITS
            or manifest.get('units') != fusion.UNITS or parameters.get('units') != fusion.UNITS
            or manifest.get('filter_run') != args.filter_run
            or manifest.get('camera_run') != first['camera_run']
            or manifest.get('selected_train_frames') != first['selected_train_frames']
            or any(manifest.get('input_and_source_hashes', {}).get(path) != digest
                for path, digest in inputs[args.filter_run]['upstream_hashes'].items())
            or report.get('parameters') != parameters
            or not all(isinstance(parameters.get(k), (int, float)) and math.isfinite(parameters[k])
                and parameters[k] > 0 for k in ('voxel_length', 'sdf_trunc', 'depth_trunc'))):
        raise ValueError('Reference must be accepted 52-TRAIN TSDF with matching fixed parameters/provenance')
    filtered = report.get('meshes', {}).get('filtered', {})
    artifacts = filtered.get('artifacts', [])
    if not filtered.get('surface_nonempty') or not any(str(a.get('file', '')).endswith('.ply') for a in artifacts):
        raise ValueError('Reference requires a nonempty accepted filtered PLY')
    for artifact in artifacts:
        path = (reference / artifact['file']).resolve()
        if Path(artifact['file']).is_absolute() or not path.is_relative_to(reference) or fusion.sha256_file(path) != artifact['sha256']:
            raise ValueError('Reference mesh artifact path/hash mismatch')
        hashes[str(path)] = artifact['sha256']
    hashes.update({str(path): fusion.sha256_file(path) for path in paths})
    return dict(inputs=inputs, parameters=parameters, upstream_hashes=hashes,
        reference_filtered=filtered, camera_run=first['camera_run'])


def variant_parameters(reference, variant):
    scale, truncation, _ = VARIANTS[variant]
    voxel = reference['voxel_length'] * scale
    return dict(voxel_length=voxel, sdf_trunc=voxel * truncation, depth_trunc=reference['depth_trunc'])


def boundary_diagnostics(vertices, triangles):
    """Boundary means exactly one adjacent triangle; never count nonmanifold edges as holes."""
    import numpy as np
    vertices, triangles = np.asarray(vertices), np.asarray(triangles)
    count = len(vertices)
    if len(triangles) > 8_000_000 or count > 2**31 - 1:
        return dict(status='unavailable', reason='Conservative boundary-analysis memory cap exceeded')
    encoded = np.concatenate([np.minimum(triangles[:, a], triangles[:, b]).astype(np.int64) * count +
        np.maximum(triangles[:, a], triangles[:, b]) for a, b in ((0, 1), (1, 2), (2, 0))])
    keys, adjacency = np.unique(encoded, return_counts=True)
    del encoded
    boundary = keys[adjacency == 1]
    nonmanifold = int((adjacency > 2).sum())
    del keys, adjacency
    ends = np.column_stack((boundary // count, boundary % count)) if count else np.empty((0, 2), np.int64)
    area = length = 0.
    for start in range(0, len(triangles), 100_000):
        points = vertices[triangles[start:start + 100_000]]
        area += float(np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1).sum() / 2)
    for start in range(0, len(ends), 100_000):
        edges = ends[start:start + 100_000]
        length += float(np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1).sum())
    active, inverse, degree = np.unique(ends.ravel(), return_inverse=True, return_counts=True)
    parents = np.arange(len(active))
    sizes = np.ones(len(active), dtype=np.int64)
    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    for a, b in inverse.reshape(-1, 2):
        a, b = find(a), find(b)
        if a != b:
            if sizes[a] < sizes[b]: a, b = b, a
            parents[b] = a
            sizes[a] += sizes[b]
    roots = np.asarray([find(i) for i in range(len(active))], dtype=np.int64)
    component_roots = np.unique(roots)
    noncycle = np.unique(roots[degree != 2])
    branched = np.unique(roots[degree > 2])
    return dict(status='passed', boundary_edge_count=len(ends), nonmanifold_edge_count=nonmanifold,
        boundary_length=length, surface_area=area, boundary_component_count=len(component_roots),
        closed_boundary_loop_count=len(component_roots) - len(noncycle),
        branched_boundary_component_count=len(branched),
        open_boundary_chain_component_count=len(noncycle) - len(branched),
        boundary_length_over_sqrt_area=length / math.sqrt(area) if area > 0 else None,
        boundary_length_over_area=length / area if area > 0 else None,
        length_units=fusion.UNITS, area_units='squared relative COLMAP world units',
        interpretation='Open mesh boundaries include scene perimeter and openings; not a hole or accuracy score')


def execute(root, run, args):
    started = time.monotonic()
    config_path = root / args.config
    config = fusion.load_yaml(config_path)
    if config.get('profile') != 'server' or Path(config['paths']['project_root']).resolve() != root:
        raise ValueError('Ablation requires declared server project root')
    if Path(sys.prefix).resolve() != (root / '.venv-geometry').resolve():
        raise ValueError('Use isolated .venv-geometry/bin/python')
    if 'open3d' in sys.modules: raise ValueError('Open3D imported before CPU isolation gate')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '4'
    sys.dont_write_bytecode = True
    resources = fusion.cpu_guard(run)
    import numpy as np
    version = importlib.metadata.version('open3d')
    if np.__version__ != '1.26.4' or version != '0.19.0':
        raise ValueError(f'Validated runtime requires NumPy 1.26.4/Open3D 0.19.0; found {np.__version__}/{version}')
    import open3d as o3d
    if o3d.__version__ != '0.19.0': raise ValueError('Imported backend must be Open3D 0.19.0')
    prepared = prepare_ablation(root, args)
    q = VARIANTS[args.variant][2]
    filter_run = args.no_confidence_filter_run if q == 0 else args.filter_run
    inputs = prepared['inputs'][filter_run]
    parameters = variant_parameters(prepared['parameters'], args.variant)
    hashes = prepared['upstream_hashes']
    for path in (config_path, Path(__file__).resolve(), root / 'scripts/fuse_tsdf.py',
            root / 'src/room_reconstruction/meshing.py', root / 'src/room_reconstruction/camera_geometry.py',
            root / 'src/room_reconstruction/filtering.py'):
        hashes[str(path)] = fusion.sha256_file(path)
    fusion.write_json(run / 'manifest.json', dict(stage=STAGE, source=fusion.git_snapshot(root), arguments=vars(args),
        variant=args.variant, confidence_quantile=q, filter_run=filter_run, camera_run=prepared['camera_run'],
        selected_train_frames=inputs['selected_train_frames'], input_and_source_hashes=hashes,
        parameters=parameters, fixed_reference_parameters=prepared['parameters'], units=fusion.UNITS,
        cpu_threads=4, backend='Open3D CPU ScalableTSDFVolume RGB8', depth_prediction_rerun=False))
    from room_reconstruction.meshing import integrate_mesh
    print(f'Integrating {args.variant}: 52 TRAIN frames; q={q}; CPU threads=4', flush=True)
    mesh = integrate_mesh(fusion.iter_frames(inputs, 'mask'), **parameters)
    fusion.check_rss()
    surface = fusion.export_mesh(mesh, run, args.variant)
    surface['boundary_diagnostics'] = boundary_diagnostics(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    del mesh
    fusion.verify_hashes(hashes)
    return dict(stage=STAGE, status='ablation_passed', requested_variants=[args.variant], completed_variants=[args.variant],
        frame_count=52, reference_run=args.reference_run, reference_mesh='filtered', reference_reused=True,
        variant=args.variant, confidence_quantile=q, filter_run=filter_run, parameters=parameters, mesh=surface,
        fixed_reference_parameters=prepared['parameters'], units=fusion.UNITS, resources=resources,
        runtime_versions=dict(numpy=np.__version__, open3d=version), host_peak_rss_bytes=fusion.check_rss(),
        elapsed_seconds=time.monotonic() - started, geometry_accuracy_validated=False, full_pipeline_ready=False,
        component_policy='All components retained; no component pruning, hole filling, or smoothing')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--filter-run', default='l2bp3_filter_002')
    parser.add_argument('--no-confidence-filter-run', required=True)
    parser.add_argument('--reference-run', default='l2bp3_tsdf_001')
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--config', default='configs/paths.server.yaml')
    args = parser.parse_args(argv)
    try: run = fusion.create_run_dir(ROOT.resolve(), args.run_id)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    try:
        report = execute(ROOT.resolve(), run, args)
        fusion.write_json(run / 'report.json', report)
        fusion.write_json(run / 'status.json', dict(stage=STAGE, status='ablation_passed', mesh_ready=True,
            requested_variants=[args.variant], completed_variants=[args.variant], geometry_accuracy_validated=False))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        failure = dict(stage=STAGE, status='failed', reason=str(exc), error_type=type(exc).__name__,
            requested_variants=[args.variant], completed_variants=[], mesh_ready=False, arguments=vars(args))
        for name in ('manifest', 'report', 'status'):
            if not (run / f'{name}.json').exists(): fusion.write_json(run / f'{name}.json', failure)
        print(f'Ablation failed: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__': raise SystemExit(main())
