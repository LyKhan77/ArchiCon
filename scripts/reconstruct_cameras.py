#!/usr/bin/env python3
"""CPU camera pilot from immutable audited train frames; no model inference."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from room_reconstruction.pipeline_layout import resolve_run_dir
from room_reconstruction.provenance import create_run_dir, git_snapshot, load_yaml, sha256_file, write_json
from room_reconstruction.resources import ResourcePolicy, evaluate_resources, parse_mem_available


def progress(run, message):
    line = f'{datetime.now(timezone.utc).isoformat()} {message}'
    with (run / 'progress.log').open('a') as stream:
        stream.write(line + '\n')
    print(line, flush=True)


def run_pilot(root, run, args):
    started = time.monotonic()
    config = load_yaml(root / args.config)
    if config.get('profile') != 'server' or Path(config['paths']['project_root']).resolve() != root:
        raise ValueError('Camera pilot requires the declared server project root')
    policy = ResourcePolicy.from_mapping(config.get('compute', {}))
    if not 1 <= args.threads <= min(4, policy.cpu_threads):
        raise ValueError('Threads exceed configured CPU resource policy')
    try:
        memory = parse_mem_available(Path('/proc/meminfo').read_text())
    except (OSError, ValueError):
        memory = None
    resources = evaluate_resources(gpus=[], policy=policy, selector=None, mem_available_bytes=memory,
        disk_free_bytes=shutil.disk_usage(run).free, busy_uuids=None, gpu_required=False)
    write_json(run / 'resources.json', resources)
    if not resources['resource_ready']:
        raise ValueError('; '.join(resources['errors']))
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', args.audit_run):
        raise ValueError('audit-run must be a run ID within outputs')
    audit = resolve_run_dir(root,args.audit_run)
    if not audit.is_relative_to(root / 'outputs'):
        raise ValueError('Audit run escapes outputs')
    audit_status_path = audit / 'status.json'
    try:
        audit_status = json.loads(audit_status_path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError('Camera pilot requires a completed video_audit status.json') from exc
    if (not isinstance(audit_status, dict) or audit_status.get('stage') != 'video_audit'
            or audit_status.get('status') != 'complete'):
        raise ValueError('Camera pilot requires a completed video_audit status.json')
    manifest_path = audit / 'frame_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    from room_reconstruction.poses import (select_pilot_frames, copy_pilot_images,
        observation_residuals, quality_summary, canonical_cameras)
    frames = select_pilot_frames(manifest['frames'], args.start, args.end)
    write_json(run / 'manifest.json', {'stage': 'camera_pilot', 'source': git_snapshot(root),
        'audit_run': args.audit_run, 'audit_manifest_sha256': sha256_file(manifest_path),
        'audit_status_sha256': sha256_file(audit_status_path),
        'source_sha256': manifest['source_sha256'], 'arguments': vars(args), 'selected_frames': frames,
        'validation_poses': 'unavailable; localization pending', 'nvs_metrics': 'unavailable'})
    progress(run, f'Copying {len(frames)} train frames at native resolution')
    copy_pilot_images(audit, run / 'images', frames)
    import numpy as np
    import pycolmap as p
    if p.__version__ != '4.2.1':
        raise ValueError(f'Validated PyCOLMAP API requires 4.2.1; found {p.__version__}')
    extraction = p.FeatureExtractionOptions()
    extraction.max_image_size = args.max_image_size
    extraction.num_threads, extraction.use_gpu = args.threads, False
    extraction.sift.max_num_features = args.max_features
    matching = p.FeatureMatchingOptions()
    matching.num_threads, matching.use_gpu = args.threads, False
    reader = p.ImageReaderOptions()
    reader.camera_model = args.camera_model
    mapping = p.IncrementalPipelineOptions()
    mapping.num_threads, mapping.mapper.num_threads = args.threads, args.threads
    mapping.ba_use_gpu, mapping.min_model_size = False, 3
    mapping.ba_local_backend = p.BundleAdjustmentBackend.CERES
    mapping.ba_global_backend = p.BundleAdjustmentBackend.CERES
    mapping.ba_local_max_num_iterations, mapping.ba_global_max_num_iterations = 25, 50
    mapping.ba_global_max_refinements, mapping.random_seed = 2, 0
    database, images, sparse = run / 'database.db', run / 'images', run / 'sparse'
    sparse.mkdir()
    progress(run, 'CPU feature extraction; shared single camera')
    p.extract_features(database, images, camera_mode=p.CameraMode.SINGLE, reader_options=reader,
                       extraction_options=extraction, device=p.Device.cpu)
    progress(run, 'CPU exhaustive matching')
    p.match_exhaustive(database, matching_options=matching, device=p.Device.cpu)
    progress(run, 'CPU incremental mapping and bundle adjustment')
    models = p.incremental_mapping(database, images, sparse, options=mapping)
    candidate_models = [{'model_id': int(i), 'registered_images': len(m.reg_image_ids()),
                   'registration_rate': len(m.reg_image_ids()) / len(frames),
                   'points3D': len(m.points3D)} for i, m in models.items()]
    candidate_models.sort(key=lambda row: (-row['registered_images'], row['model_id']))
    report = {'stage': 'camera_pilot', 'selected_train_frames': len(frames), 'candidate_models': candidate_models,
        'candidate_model_count': len(candidate_models), 'connected_component_count': None,
        'connectivity_note': 'Candidate models may overlap; graph connectivity not separately measured',
        'primary_model_id': None, 'N_largest': 0,
        'quality': quality_summary([], 0, len(frames)), 'camera_model': args.camera_model,
        'runtime_versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pycolmap': p.__version__},
        'validation_poses': 'unavailable; localization pending', 'nvs_metrics': 'unavailable',
        'undistorted': False, 'target_train_count': [45, 55],
        'target_train_count_met': 45 <= len(frames) <= 55}
    if candidate_models:
        primary_id = candidate_models[0]['model_id']
        model = models[primary_id]
        primary = run / 'primary'
        primary.mkdir()
        model.write(primary)
        cameras = canonical_cameras(model, frames)
        write_json(run / 'cameras.json', cameras)
        write_json(run / 'trajectory.json', {'world': cameras['world'], 'pose_convention': cameras['pose_convention'],
            'frames': [{key: row[key] for key in ('frame_id', 'timestamp_seconds', 'c2w')} for row in cameras['frames']]})
        report.update(primary_model_id=primary_id, N_largest=len(model.reg_image_ids()),
                      quality=quality_summary(observation_residuals(model), len(model.reg_image_ids()), len(frames)))
        # Persist sparse results before the optional undistortion stage.
        write_json(run / 'sparse_report.json', report)
        if report['quality']['passed']:
            progress(run, 'Camera gate passed; undistorting train images to pinhole model')
            undistorted = run / 'undistorted'
            p.undistort_images(undistorted, primary, images, output_type='COLMAP', num_threads=args.threads)
            pinhole = p.Reconstruction(undistorted / 'sparse')
            # Undistortion can change dimensions and K; export its own metadata.
            native = {row['image_name']: row for row in cameras['frames']}
            undistorted_frames = []
            for image_id in pinhole.reg_image_ids():
                image = pinhole.images[image_id]
                camera = pinhole.cameras[image.camera_id]
                if camera.model_name not in ('PINHOLE', 'SIMPLE_PINHOLE'):
                    raise ValueError('Undistorter did not produce a pinhole camera')
                row = dict(native[image.name])
                row.update(image_size=[camera.width, camera.height], camera_model=camera.model_name,
                    camera_params=camera.params.tolist(), K=camera.calibration_matrix().tolist(),
                    image_sha256=sha256_file(undistorted / 'images' / image.name))
                undistorted_frames.append(row)
            write_json(undistorted / 'cameras.json', dict(cameras, frames=undistorted_frames,
                pixel_units='undistorted image pixels', image_root='images'))
            report['undistorted'] = True
    report['elapsed_seconds'] = time.monotonic() - started
    write_json(run / 'report.json', report)
    progress(run, f"Finished: largest={report['N_largest']}/{len(frames)} gate={report['quality']['passed']}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-run', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--config', default='configs/paths.server.yaml')
    parser.add_argument('--start', type=float, default=15)
    parser.add_argument('--end', type=float, default=45)
    parser.add_argument('--camera-model', choices=['OPENCV', 'SIMPLE_RADIAL', 'PINHOLE'], default='OPENCV')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-image-size', type=int, default=960)
    parser.add_argument('--max-features', type=int, default=4096)
    args = parser.parse_args(argv)
    try:
        run = create_run_dir(ROOT.resolve(), args.run_id)
    except (OSError, ValueError) as exc:
        print(f'camera pilot: {exc}', file=sys.stderr)
        return 2
    try:
        if min(args.threads, args.max_image_size, args.max_features) <= 0:
            raise ValueError('Threads, max-image-size and max-features must be positive')
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[key] = str(args.threads)
        report = run_pilot(ROOT.resolve(), run, args)
        passed = report['quality']['passed']
        write_json(run / 'status.json', {'status': 'camera_passed' if passed else 'camera_rejected',
            'camera_ready': passed, 'inference_ready': False})
        return 0 if passed else 2
    except (Exception, KeyboardInterrupt) as exc:
        write_json(run / 'status.json', {'status': 'failed', 'reason': str(exc),
            'error_type': type(exc).__name__, 'camera_ready': False, 'inference_ready': False})
        print(f'camera pilot failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
