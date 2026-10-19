#!/usr/bin/env python3
"""CPU held-out localization; no mapping, bundle adjustment or scene optimization."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
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


def input_run(root, name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', name):
        raise ValueError('Input run must be a run ID within outputs')
    path = resolve_run_dir(root,name)
    if not path.is_relative_to(root / 'outputs'):
        raise ValueError('Input run escapes outputs')
    return path


def run_localization(root, run, args):
    started = time.monotonic()
    config = load_yaml(root / args.config)
    if config.get('profile') != 'server' or Path(config['paths']['project_root']).resolve() != root:
        raise ValueError('Localization requires the declared server project root')
    policy = ResourcePolicy.from_mapping(config.get('compute', {}))
    if not 1 <= args.threads <= min(4, policy.cpu_threads):
        raise ValueError('Threads exceed configured CPU resource policy')
    # Even a looser config cannot lower the held-out localization resource gates.
    policy = ResourcePolicy(min_host_available_gib=max(8, policy.min_host_available_gib),
                            min_disk_free_gib=max(50, policy.min_disk_free_gib), cpu_threads=args.threads)
    try:
        memory = parse_mem_available(Path('/proc/meminfo').read_text())
    except (OSError, ValueError):
        memory = None
    resources = evaluate_resources(gpus=[], policy=policy, selector=None, mem_available_bytes=memory,
        disk_free_bytes=shutil.disk_usage(run).free, busy_uuids=None, gpu_required=False)
    write_json(run / 'resources.json', resources)
    if not resources['resource_ready']:
        raise ValueError('; '.join(resources['errors']))
    from room_reconstruction import localization as loc
    from room_reconstruction.poses import copy_pilot_images
    audit, training = input_run(root, args.audit_run), input_run(root, args.camera_run)
    manifest_path = audit / 'frame_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    pilot_manifest = json.loads((training / 'manifest.json').read_text())
    loc.validate_upstream(json.loads((audit / 'status.json').read_text()),
        json.loads((training / 'status.json').read_text()), pilot_manifest, args.audit_run, sha256_file(manifest_path))
    frames = loc.select_validation_frames(manifest['frames'], args.start, args.end)
    before = loc.map_snapshot(training)
    write_json(run / 'map_hashes_before.json', before)
    write_json(run / 'manifest.json', {'stage': 'validation_localization', 'source': git_snapshot(root),
        'audit_run': args.audit_run, 'camera_run': args.camera_run, 'arguments': vars(args),
        'audit_manifest_sha256': sha256_file(manifest_path), 'source_sha256': manifest['source_sha256'],
        'selected_validation_frames': frames, 'map_hashes_before': before,
        'protocol': 'Query-only fixed-map absolute pose; no map BA, query triangulation or scene optimizer'})
    try:
        progress(run, 'Verifying and copying eight held-out native images')
        copy_pilot_images(audit, run / 'images', frames)
        import numpy as np
        import pycolmap as p
        if p.__version__ != '4.2.1':
            raise ValueError(f'Validated PyCOLMAP API requires 4.2.1; found {p.__version__}')
        model = p.Reconstruction(training / 'primary')
        train_images = [model.images[i] for i in model.reg_image_ids()]
        loc.validate_training_names([image.name for image in train_images], manifest['frames'])
        used_cameras = {image.camera_id for image in train_images}
        if len(used_cameras) != 1:
            raise ValueError('Localization requires one shared fixed training camera')
        camera = model.cameras[next(iter(used_cameras))]
        if any(row['display_size'] != [camera.width, camera.height] for row in frames):
            raise ValueError('Validation dimensions differ from native training camera')
        database = run / 'database.db'
        # SQLite backup handles journals and leaves the source connection read-only.
        with sqlite3.connect(f'{(training / "database.db").as_uri()}?mode=ro', uri=True) as source:
            with sqlite3.connect(database) as destination:
                source.backup(destination)
        with p.Database.open(database) as db:
            db_images = db.read_all_images()
            db_images = list(db_images.values()) if isinstance(db_images, dict) else list(db_images)
            loc.validate_training_names([image.name for image in db_images], manifest['frames'])
            for image in train_images:
                stored = db.read_image_with_name(image.name)
                if stored.image_id != image.image_id or len(db.read_keypoints(image.image_id)) != len(image.points2D):
                    raise ValueError('Training model tracks differ from database keypoints')
            # Pilot DB contains the initial camera. Replace only the copied DB calibration
            # with the refined camera to make geometric verification use the fixed map K.
            db.update_camera(camera)
        extraction = p.FeatureExtractionOptions()
        extraction.num_threads, extraction.use_gpu = args.threads, False
        extraction.max_image_size = pilot_manifest['arguments']['max_image_size']
        extraction.sift.max_num_features = pilot_manifest['arguments']['max_features']
        reader = p.ImageReaderOptions()
        reader.existing_camera_id = camera.camera_id
        names = [Path(row['image_path']).name for row in frames]
        progress(run, 'CPU query feature extraction into copied training database')
        p.extract_features(database, run / 'images', image_names=names, reader_options=reader,
                           extraction_options=extraction, device=p.Device.cpu)
        pairs = run / 'query_train_pairs.txt'
        with pairs.open('x') as stream:
            for name in names:
                for image in train_images:
                    stream.write(f'{name} {image.name}\n')
        matching = p.FeatureMatchingOptions()
        matching.num_threads, matching.use_gpu = args.threads, False
        pairing = p.ImportedPairingOptions()
        pairing.match_list_path = str(pairs)
        progress(run, f'CPU geometric verification of {len(names) * len(train_images)} query–train pairs')
        p.match_image_pairs(database, matching_options=matching, pairing_options=pairing, device=p.Device.cpu)
        points = {pid: point.xyz for pid, point in model.points3D.items()}
        records, poses = [], []
        with p.Database.open(database) as db:
            for row, name in zip(frames, names, strict=True):
                query = db.read_image_with_name(name)
                votes, verified_count = [], 0
                for train in train_images:
                    geometry = db.read_two_view_geometry(query.image_id, train.image_id)
                    for query_index, train_index in geometry.inlier_matches:
                        verified_count += 1
                        observation = train.points2D[int(train_index)]
                        if observation.has_point3D():
                            votes.append((int(query_index), int(observation.point3D_id)))
                pixels, xyz, indices = loc.resolve_correspondences(db.read_keypoints(query.image_id), votes, points)
                pose = loc.estimate_query_pose(pixels, xyz, camera, p)
                records.append(dict(frame_id=row['frame_id'], image_name=name, verified_match_count=verified_count,
                                    track_vote_count=len(votes), **pose))
                write_json(run / f'correspondences_{row["frame_id"]}.json', {
                    'frame_id': row['frame_id'], 'query_keypoint_and_map_point_ids': indices,
                    'points2D_native_pixels': pixels.tolist(), 'points3D_fixed_world': xyz.tolist()})
                if pose['quality']['passed']:
                    poses.append({'frame_id': row['frame_id'], 'source_index': row['source_index'],
                        'timestamp_seconds': row['timestamp_seconds'], 'image_name': name,
                        'image_sha256': row['image_sha256'], 'split': 'validation', 'camera_id': camera.camera_id,
                        'image_size': [camera.width, camera.height], 'camera_model': camera.model_name,
                        'camera_params': camera.params.tolist(), 'K': camera.calibration_matrix().tolist(), 'c2w': pose['c2w']})
                progress(run, f'{name}: {pose["quality"]["inlier_count"]}/{len(indices)} inliers; passed={pose["quality"]["passed"]}')
        cameras = {'schema_version': 1, 'pose_convention': 'OpenCV c2w; +x right, +y down, +z forward',
                   'world': 'Relative, unscaled fixed COLMAP training world', 'pixel_units': 'native image pixels',
                   'image_root': 'images', 'frames': poses, 'calibration_fixed': True, 'training_map_fixed': True}
        write_json(run / 'cameras.json', cameras)
        quality = loc.localization_quality([row['quality'] for row in records])
        undistorted = loc.undistort_queries(run, training, cameras, threads=args.threads) if quality['passed'] else {
            'available': False, 'reason': 'Localization gate rejected; native accepted cameras only'}
        after = loc.map_snapshot(training)
        report = {'stage': 'validation_localization', 'quality': quality, 'queries': records,
            'map_unchanged': before == after, 'map_hashes_before': before, 'map_hashes_after': after,
            'calibration_fixed': True, 'training_map_fixed': True, 'bundle_adjustment_run': False,
            'scene_optimizer_run': False, 'query_triangulation_run': False, 'undistorted': undistorted,
            'residual_definition': 'Native distorted camera Euclidean residual on absolute pose RANSAC inliers',
            'held_out_use': 'RGB used only for query feature extraction/localization and optional undistortion',
            'runtime_versions': {'pycolmap': p.__version__, 'numpy': np.__version__},
            'elapsed_seconds': time.monotonic() - started, 'nvs_metrics': 'unavailable'}
        if not report['map_unchanged']:
            raise ValueError('Training map/database changed during localization')
        write_json(run / 'report.json', report)
        return report
    finally:
        after = loc.map_snapshot(training)
        write_json(run / 'map_hashes_after.json', {'files': after, 'unchanged': before == after})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-run', required=True)
    parser.add_argument('--camera-run', default='l2bp3_camera_pilot_002')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--config', default='configs/paths.server.yaml')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--start', type=float, default=15)
    parser.add_argument('--end', type=float, default=45)
    args = parser.parse_args(argv)
    try:
        run = create_run_dir(ROOT.resolve(), args.run_id)
    except (OSError, ValueError) as exc:
        print(f'validation localization: {exc}', file=sys.stderr)
        return 2
    try:
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[key] = str(args.threads)
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        report = run_localization(ROOT.resolve(), run, args)
        passed = report['quality']['passed'] and report['map_unchanged']
        write_json(run / 'status.json', {'stage': 'validation_localization',
            'status': 'localization_passed' if passed else 'localization_rejected',
            'validation_camera_ready': passed, 'camera_ready': passed, 'inference_ready': False})
        return 0 if passed else 2
    except (Exception, KeyboardInterrupt) as exc:
        write_json(run / 'status.json', {'stage': 'validation_localization', 'status': 'failed',
            'reason': str(exc), 'error_type': type(exc).__name__, 'validation_camera_ready': False,
            'camera_ready': False, 'inference_ready': False})
        print(f'validation localization failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
