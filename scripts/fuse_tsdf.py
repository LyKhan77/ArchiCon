#!/usr/bin/env python3
"""CPU TSDF baseline/filtered surfaces in relative COLMAP units; no accuracy claim."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import resource
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from room_reconstruction.pipeline_layout import resolve_run_dir
from room_reconstruction.provenance import create_run_dir, git_snapshot, load_yaml, sha256_file, write_json
from room_reconstruction.resources import GIB, ResourcePolicy, evaluate_resources, parse_mem_available

UNITS = 'relative COLMAP world units; not meters'
STAGE = 'tsdf_mesh'


def cpu_guard(run):
    resources = evaluate_resources(gpus=[], policy=ResourcePolicy(), selector=None,
        mem_available_bytes=parse_mem_available(Path('/proc/meminfo').read_text()),
        disk_free_bytes=shutil.disk_usage(run).free, busy_uuids=None, gpu_required=False)
    write_json(run / 'resources.json', resources)
    if not resources['resource_ready']:
        raise ValueError('; '.join(resources['errors']))
    return resources


def check_rss():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    if peak > 6 * GIB:
        raise ValueError('TSDF process exceeded 6 GiB host RSS budget')
    return peak


def run_path(root, name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', name):
        raise ValueError('Unsafe input run ID')
    path = resolve_run_dir(root,name)
    if not path.is_relative_to((root / 'outputs').resolve()):
        raise ValueError('Input run escapes outputs')
    return path


def camera_inputs(path):
    from room_reconstruction.depth_pilot import prepare_inputs
    return prepare_inputs(path)


def prepare_filter(root, name):
    run = run_path(root, name)
    paths = [run / f'{key}.json' for key in ('status', 'manifest', 'report')]
    if any(not p.resolve().is_relative_to(run) for p in paths):
        raise ValueError('Filter metadata escapes run')
    status, manifest, report = [json.loads(p.read_text()) for p in paths]
    if (status.get('stage') != 'filtered_depth' or status.get('status') != 'filter_passed'
            or manifest.get('stage') != 'filtered_depth' or report.get('stage') != 'filtered_depth'
            or report.get('frame_count') != 52 or report.get('units') != UNITS
            or manifest.get('camera_run') != report.get('camera_run')):
        raise ValueError('A complete 52-frame filter_passed relative-depth stage is required')
    selected = manifest.get('selected_train_frames', [])
    rows = report.get('frames', [])
    ids = [r['frame_id'] for r in rows]
    if (len(selected) != 52 or len(rows) != 52 or len(set(ids)) != 52
            or any(r.get('split') != 'train' for r in selected)
            or {r['frame_id'] for r in selected} != set(ids)):
        raise ValueError('Exactly 52 unique TRAIN artifacts required')
    camera = camera_inputs(run_path(root, report['camera_run']))
    camera_rows = {r['frame_id']: r for r in camera['frames']}
    if set(camera_rows) != set(ids) or any(r != camera_rows[r['frame_id']] for r in selected):
        raise ValueError('Filter artifacts differ from accepted camera TRAIN frames')
    hashes = {str(p): sha256_file(p) for p in paths}
    hashes.update(camera['upstream_hashes'])
    for name, digest in manifest.get('upstream_hashes', {}).items():
        path = Path(name).resolve()
        if not path.is_relative_to((root / 'outputs').resolve()) or sha256_file(path) != digest:
            raise ValueError('Filter upstream path/hash mismatch')
        hashes[str(path)] = digest
    artifacts = []
    for row in rows:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', row['frame_id']):
            raise ValueError('Unsafe frame ID')
        file = row['file']
        path = (run / file).resolve()
        if Path(file).is_absolute() or not path.is_relative_to(run) or not path.is_file() or sha256_file(path) != row['sha256']:
            raise ValueError('Filter artifact path/hash mismatch')
        artifacts.append(path)
        hashes[str(path)] = row['sha256']
    if len(set(artifacts)) != 52:
        raise ValueError('Duplicate filter artifact paths')
    return dict(paths=artifacts, frames=rows, selected_train_frames=selected,
        upstream_hashes=hashes, camera_run=report['camera_run'], filter_report=report)


def iter_frames(inputs, mask_key):
    import numpy as np
    from room_reconstruction.depth_da3 import expected_resize
    from room_reconstruction.meshing import integration_arrays
    selected = {row['frame_id']: row for row in inputs['selected_train_frames']}
    cameras = [selected[row['frame_id']] for row in inputs['frames']]
    expected_K, shape = expected_resize(np.asarray([row['K'] for row in cameras]), cameras[0]['image_size'])
    expected_E = np.linalg.inv(np.asarray([row['c2w'] for row in cameras]))[:, :3]
    for index, path in enumerate(inputs['paths']):
        check_rss()
        with np.load(path, allow_pickle=False) as archive:
            frame = {key: archive[key] for key in ('depth', 'conf', 'processed_rgb', 'K', 'w2c',
                'base_mask', 'mask', 'normal', 'normal_valid')}
        depth = frame['depth']
        if (depth.shape != shape or frame['K'].shape != (3, 3) or frame['w2c'].shape != (3, 4)
                or not np.allclose(frame['K'], expected_K[index], rtol=0, atol=2e-4)
                or not np.allclose(frame['w2c'], expected_E[index], rtol=0, atol=2e-5)):
            raise ValueError('Filtered camera differs from accepted global TRAIN camera')
        if (depth.dtype != np.float32 or depth.ndim != 2 or frame['conf'].shape != depth.shape
                or frame['normal'].shape != (*depth.shape, 3) or frame['normal'].dtype != np.float32
                or not np.isfinite(frame['normal']).all()):
            raise ValueError('Invalid filtered depth/confidence/normal arrays')
        for key in ('base_mask', 'mask', 'normal_valid'):
            if frame[key].dtype != bool or frame[key].shape != depth.shape:
                raise ValueError(f'Invalid {key} boolean mask')
        if (np.any(frame['mask'] & ~frame['base_mask']) or np.any(frame['normal_valid'] & ~frame['mask'])
                or np.any(frame['base_mask'] & (~np.isfinite(depth) | (depth <= 0) | ~np.isfinite(frame['conf'])))):
            raise ValueError('Invalid mask membership or masked depth/confidence')
        integration_arrays(frame, frame[mask_key], float(np.finfo(np.float32).max))
        yield frame, frame[mask_key]


def scene_parameters(inputs):
    import numpy as np
    samples = []
    for frame, mask in iter_frames(inputs, 'base_mask'):
        values = frame['depth'][mask]
        if len(values):
            samples.append(values[::max(1, (len(values) + 4095) // 4096)].copy())
    if not samples:
        raise ValueError('No valid baseline depths for TSDF parameters')
    values = np.concatenate(samples)
    median = float(np.median(values))
    return dict(voxel_length=median / 150, sdf_trunc=4 * median / 150,
        depth_trunc=1.1 * float(np.quantile(values, .995)), median_sample_depth=median,
        sampled_depth_count=len(values), maximum_samples_per_frame=4096,
        units=UNITS, rule='Deterministic bounded base-mask sample, shared by both integrations')


def export_mesh(mesh, run, name):
    import numpy as np
    import open3d as o3d
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)
    if (not len(vertices) or not len(triangles) or not np.isfinite(vertices).all()
            or colors.shape != vertices.shape or not np.isfinite(colors).all()
            or np.any((colors < 0) | (colors > 1))):
        raise ValueError(f'{name} surface is empty, nonfinite, or lacks valid RGB colors')
    _, counts, areas = mesh.cluster_connected_triangles()
    stats = dict(vertex_count=len(vertices), triangle_count=len(triangles), surface_nonempty=True,
        finite_vertices=True, bounds=[vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        component_count=len(counts), largest_component_triangle_counts=sorted(map(int, counts), reverse=True)[:20],
        component_surface_area_sum=float(sum(areas)),
        nonmanifold_edge_count=len(mesh.get_non_manifold_edges(allow_boundary_edges=True)),
        boundary_or_nonmanifold_edge_count=len(mesh.get_non_manifold_edges(allow_boundary_edges=False)),
        vertex_manifold=bool(mesh.is_vertex_manifold()), all_components_retained=True)
    artifacts = []
    ply = run / f'{name}.ply'
    if ply.exists() or not o3d.io.write_triangle_mesh(str(ply), mesh, write_vertex_colors=True):
        raise ValueError('PLY export failed or would overwrite an artifact')
    artifacts.append(dict(file=ply.name, sha256=sha256_file(ply), size_bytes=ply.stat().st_size))
    try:
        import trimesh
        glb = run / f'{name}.glb'
        surface = trimesh.Trimesh(vertices=vertices, faces=triangles,
            vertex_colors=np.rint(colors * 255).astype(np.uint8), process=False)
        with glb.open('xb') as stream:
            stream.write(surface.export(file_type='glb'))
        artifacts.append(dict(file=glb.name, sha256=sha256_file(glb), size_bytes=glb.stat().st_size))
        stats['glb_export'] = 'passed'
    except (ImportError, ValueError, TypeError, AttributeError) as exc:
        stats['glb_export'] = f'unavailable: {type(exc).__name__}: {exc}'
    return dict(stats, artifacts=artifacts)


def verify_hashes(hashes):
    for name, digest in hashes.items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f'Input or source changed during fusion: {name}')


def run_fusion(root, run, args):
    started = time.monotonic()
    config_path = root / args.config
    config = load_yaml(config_path)
    if config.get('profile') != 'server' or Path(config['paths']['project_root']).resolve() != root:
        raise ValueError('TSDF requires the declared server project root')
    if Path(sys.prefix).resolve() != (root / '.venv-geometry').resolve():
        raise ValueError('Use isolated .venv-geometry/bin/python')
    if 'open3d' in sys.modules:
        raise ValueError('Open3D imported before CPU isolation gate')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '4'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    resources = cpu_guard(run)
    import numpy as np
    if np.__version__ != '1.26.4':
        raise ValueError(f'Validated runtime requires NumPy 1.26.4; found {np.__version__}')
    version = importlib.metadata.version('open3d')
    if version != '0.19.0':
        raise ValueError(f'Validated runtime requires Open3D 0.19.0; found {version}')
    import open3d as o3d
    if o3d.__version__ != '0.19.0':
        raise ValueError(f'Imported backend must be Open3D 0.19.0; found {o3d.__version__}')
    inputs = prepare_filter(root, args.filter_run)
    parameters = scene_parameters(inputs)
    source = git_snapshot(root)
    hashes = dict(inputs['upstream_hashes'])
    for path in [config_path, root / 'scripts/fuse_tsdf.py', root / 'src/room_reconstruction/meshing.py',
            root / 'src/room_reconstruction/camera_geometry.py']:
        hashes[str(path)] = sha256_file(path)
    write_json(run / 'manifest.json', dict(stage=STAGE, source=source, arguments=vars(args),
        filter_run=args.filter_run, camera_run=inputs['camera_run'], selected_train_frames=inputs['selected_train_frames'],
        input_and_source_hashes=hashes, parameters=parameters, units=UNITS, cpu_threads=4,
        backend='Open3D CPU ScalableTSDFVolume RGB8', ground_truth=False, geometry_accuracy_validated=False))
    from room_reconstruction.meshing import integrate_mesh
    meshes = {}
    settings = {key: parameters[key] for key in ('voxel_length', 'sdf_trunc', 'depth_trunc')}
    for name, mask in [('baseline', 'base_mask'), ('filtered', 'mask')]:
        print(f'Integrating {name}: 52 TRAIN frames, relative scene units, CPU threads=4', flush=True)
        mesh = integrate_mesh(iter_frames(inputs, mask), **settings)
        check_rss()
        meshes[name] = export_mesh(mesh, run, name)
        del mesh
    verify_hashes(hashes)
    return dict(stage=STAGE, status='mesh_passed', frame_count=52, meshes=meshes, parameters=parameters,
        units=UNITS, ground_truth=False, geometry_accuracy_validated=False, full_pipeline_ready=False,
        mask_complete=inputs['filter_report'].get('mask_complete', False),
        acceptance='Both colored triangle surfaces nonempty and finite; no geometric accuracy claim',
        component_policy='All components retained; no component pruning, hole filling, or smoothing',
        runtime_versions=dict(numpy=np.__version__, open3d=version), resources=resources,
        host_peak_rss_bytes=check_rss(), elapsed_seconds=time.monotonic() - started)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--filter-run', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--config', default='configs/paths.server.yaml')
    args = parser.parse_args(argv)
    try:
        run = create_run_dir(ROOT.resolve(), args.run_id)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    try:
        report = run_fusion(ROOT.resolve(), run, args)
        write_json(run / 'report.json', report)
        write_json(run / 'status.json', dict(stage=STAGE, status='mesh_passed', mesh_ready=True,
            geometry_accuracy_validated=False, full_pipeline_ready=False))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        failure = dict(stage=STAGE, status='failed', reason=str(exc), error_type=type(exc).__name__,
            mesh_ready=False, geometry_accuracy_validated=False, full_pipeline_ready=False, arguments=vars(args))
        for name in ('manifest', 'report', 'status'):
            if not (run / f'{name}.json').exists():
                write_json(run / f'{name}.json', failure)
        print(f'TSDF failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
