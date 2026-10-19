#!/usr/bin/env python3
"""CPU SfM over every audited video frame, preserving separate component worlds."""
import argparse
from dataclasses import replace
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


def run_full(root, run, args):
    started = time.monotonic()
    config = load_yaml(root / args.config)
    if config.get('profile') != 'server' or Path(config['paths']['project_root']).resolve() != root:
        raise ValueError('Full camera reconstruction requires the declared server project root')
    policy = ResourcePolicy.from_mapping(config.get('compute', {}))
    policy = replace(policy, min_host_available_gib=max(8, policy.min_host_available_gib),
                     min_disk_free_gib=max(50, policy.min_disk_free_gib))
    if not 1 <= args.threads <= min(4, policy.cpu_threads):
        raise ValueError('Threads exceed configured CPU resource policy')
    if args.matching not in ('exhaustive', 'sequential') or args.overlap <= 0:
        raise ValueError('Matching must be exhaustive/sequential and overlap must be positive')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(args.threads)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
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
    audit_status_path, manifest_path = audit / 'status.json', audit / 'frame_manifest.json'
    try:
        status = json.loads(audit_status_path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError('Full reconstruction requires a completed video_audit status.json') from exc
    if not isinstance(status, dict) or status.get('stage') != 'video_audit' or status.get('status') != 'complete':
        raise ValueError('Full reconstruction requires a completed video_audit status.json')
    manifest = json.loads(manifest_path.read_text())
    # Heavy numerical/SfM imports remain behind the CPU resource and audit gates.
    from room_reconstruction.full_cameras import (select_full_frames, production_cameras,
                                                  component_quality, coverage_summary)
    from room_reconstruction.poses import copy_pilot_images, observation_residuals
    frames = select_full_frames(manifest['frames'])
    common = {'audit_run': args.audit_run, 'audit_manifest_sha256': sha256_file(manifest_path),
              'audit_status_sha256': sha256_file(audit_status_path), 'source_sha256': manifest['source_sha256'],
              'production_uses_original_validation': True, 'reconstruction_role': 'production',
              'matching_policy': {'mode': args.matching,
                  'overlap': args.overlap if args.matching == 'sequential' else None,
                  'quadratic_overlap': False, 'loop_detection': False, 'device': 'cpu',
                  'num_threads': args.threads},
              'evaluation_scope': 'All audit splits used as production input; no unbiased held-out evaluation',
              'nvs_metrics': 'unavailable; production input includes original audit validation images'}
    write_json(run / 'manifest.json', dict(common, stage='camera_full', source=git_snapshot(root),
        arguments=vars(args), selected_frames=frames,
        observed_timestamp_range_seconds=[frames[0]['timestamp_seconds'], frames[-1]['timestamp_seconds']]))
    progress(run, f'Copying all {len(frames)} audited observations at native resolution')
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
    progress(run, 'CPU feature extraction; shared single camera; CUDA hidden')
    p.extract_features(database, images, camera_mode=p.CameraMode.SINGLE, reader_options=reader,
                       extraction_options=extraction, device=p.Device.cpu)
    if args.matching == 'sequential':
        pairing = p.SequentialPairingOptions()
        pairing.overlap, pairing.num_threads = args.overlap, args.threads
        pairing.quadratic_overlap, pairing.loop_detection = False, False
        progress(run, f'CPU sequential matching: overlap={args.overlap}; quadratic and loop detection disabled')
        p.match_sequential(database, matching_options=matching, pairing_options=pairing, device=p.Device.cpu)
    else:
        progress(run, f'CPU exhaustive matching: {len(frames) * (len(frames) - 1) // 2} image pairs')
        p.match_exhaustive(database, matching_options=matching, device=p.Device.cpu)
    progress(run, 'CPU incremental mapping; preserve all returned reconstruction models')
    models = p.incremental_mapping(database, images, sparse, options=mapping)
    ordered = sorted(models.items(), key=lambda item: (-len(item[1].reg_image_ids()), int(item[0])))
    (run / 'components').mkdir()
    by_name = {Path(row['image_path']).name: row for row in frames}
    components = []
    for index, (model_id, model) in enumerate(ordered):
        component_id = f'component_{index:03}'
        component = run / 'components' / component_id
        primary = component / 'primary'
        primary.mkdir(parents=True)
        model.write(primary)
        selected = sorted([by_name[model.images[i].name] for i in model.reg_image_ids()],
                          key=lambda row: (row['timestamp_seconds'], row['frame_id']))
        write_json(component / 'manifest.json', dict(common, stage='camera_full_component',
            parent_run=args.run_id, component_id=component_id, model_id=int(model_id),
            selected_frames=selected, coordinate_frame_id=component_id,
            coordinate_scope='Separate relative component-local world; not aligned to other models'))
        report = {'stage': 'camera_full_component', 'component_id': component_id,
                  'model_id': int(model_id), 'coordinate_frame_id': component_id,
                  'world': 'Relative unscaled component-local COLMAP world; no inter-component alignment',
                  'registered_images': len(selected), 'frame_ids': [row['frame_id'] for row in selected],
                  'points3D': len(model.points3D), 'undistorted': False,
                  'production_uses_original_validation': True,
                  'evaluation_scope': common['evaluation_scope']}
        try:
            cameras = production_cameras(model, frames, component_id)
            cameras['image_root'] = '../../images'
            quality = component_quality(observation_residuals(model), cameras)
            write_json(component / 'cameras.json', cameras)
        except (ValueError, KeyError, TypeError) as exc:
            # Keep native model for diagnosis; invalid numeric JSON cannot enter downstream geometry.
            cameras = {'frames': [], 'coordinate_frame_id': component_id, 'export_error': str(exc)}
            quality = component_quality([], cameras)
            report['camera_export_error'] = str(exc)
            write_json(component / 'cameras.json', cameras)
        report['quality'] = quality
        accepted = quality['passed']
        if accepted:
            progress(run, f'{component_id}: accepted {len(selected)} frames; undistorting independently')
            try:
                undistorted = component / 'undistorted'
                p.undistort_images(undistorted, primary, images, output_type='COLMAP', num_threads=args.threads)
                pinhole = p.Reconstruction(undistorted / 'sparse')
                native = {row['image_name']: row for row in cameras['frames']}
                pinhole_frames = []
                for image_id in pinhole.reg_image_ids():
                    image = pinhole.images[image_id]
                    camera = pinhole.cameras[image.camera_id]
                    if camera.model_name not in ('PINHOLE', 'SIMPLE_PINHOLE'):
                        raise ValueError('Undistorter did not produce a pinhole camera')
                    row = dict(native[image.name])
                    row.update(image_size=[camera.width, camera.height], camera_model=camera.model_name,
                        camera_params=camera.params.tolist(), K=camera.calibration_matrix().tolist(),
                        image_sha256=sha256_file(undistorted / 'images' / image.name))
                    pinhole_frames.append(row)
                if {row['frame_id'] for row in pinhole_frames} != set(report['frame_ids']):
                    raise ValueError('Undistortion changed registered frame membership')
                if not component_quality([0], {'frames': pinhole_frames})['calibration_valid']:
                    raise ValueError('Undistorted calibration is invalid')
                write_json(undistorted / 'cameras.json', dict(cameras,
                    frames=sorted(pinhole_frames, key=lambda row: row['timestamp_seconds']),
                    pixel_units='undistorted image pixels', image_root='images'))
                report['undistorted'] = True
            except Exception as exc:
                report['undistortion_error'] = f'{type(exc).__name__}: {exc}'
                accepted = False
        report['camera_ready'] = accepted
        write_json(component / 'report.json', report)
        write_json(component / 'status.json', {'stage': 'camera_full_component',
            'status': 'camera_passed' if accepted else 'camera_rejected', 'camera_ready': accepted,
            'inference_ready': False, 'coordinate_frame_id': component_id})
        components.append({'component_id': component_id, 'path': f'components/{component_id}',
            'model_id': int(model_id), 'accepted': accepted, 'frame_count': len(selected),
            'frame_ids': report['frame_ids'], 'quality': quality, 'undistorted': report['undistorted'],
            'coordinate_frame_id': component_id})
    coverage = coverage_summary(frames, components)
    passed = any(row['accepted'] for row in components)
    report = dict(common, stage='camera_full', components=components, candidate_model_count=len(components),
        component_count=len(components), accepted_component_count=sum(row['accepted'] for row in components),
        model_relationship='Returned models may overlap; retained in separate coordinate frames; no forced merging',
        camera_ready=passed, quality={'passed': passed, 'scope': 'At least one accepted component; coverage reported separately'},
        runtime_versions={'python': platform.python_version(), 'numpy': np.__version__, 'pycolmap': p.__version__},
        elapsed_seconds=time.monotonic() - started, **coverage)
    write_json(run / 'report.json', report)
    progress(run, f"Finished: accepted={report['accepted_component_count']} coverage={coverage['accepted_union_count']}/{len(frames)} complete={coverage['coverage_complete']}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-run', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--config', default='configs/paths.server.yaml')
    parser.add_argument('--camera-model', choices=['OPENCV', 'SIMPLE_RADIAL', 'PINHOLE'], default='OPENCV')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-image-size', type=int, default=960)
    parser.add_argument('--max-features', type=int, default=4096)
    parser.add_argument('--matching', choices=['exhaustive', 'sequential'], default='exhaustive')
    parser.add_argument('--overlap', type=int, default=16)
    args = parser.parse_args(argv)
    try:
        run = create_run_dir(ROOT.resolve(), args.run_id)
    except (OSError, ValueError) as exc:
        print(f'full cameras: {exc}', file=sys.stderr)
        return 2
    try:
        if min(args.threads, args.max_image_size, args.max_features, args.overlap) <= 0:
            raise ValueError('Threads, max-image-size, max-features and overlap must be positive')
        report = run_full(ROOT.resolve(), run, args)
        passed = report['camera_ready']
        write_json(run / 'status.json', {'stage': 'camera_full',
            'status': 'camera_passed' if passed else 'camera_rejected',
            'camera_ready': passed, 'inference_ready': False,
            'coverage_complete': report['coverage_complete'], 'partial_coverage': report['partial_coverage'],
            'accepted_union_count': report['accepted_union_count'], 'selected_frame_count': report['selected_frame_count']})
        return 0 if passed else 2
    except (Exception, KeyboardInterrupt) as exc:
        write_json(run / 'status.json', {'stage': 'camera_full', 'status': 'failed', 'reason': str(exc),
            'error_type': type(exc).__name__, 'camera_ready': False, 'inference_ready': False,
            'coverage_complete': False})
        print(f'full cameras failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
