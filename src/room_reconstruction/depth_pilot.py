"""CPU contracts for bounded, overlapping DA3 windows in one global camera frame."""
from __future__ import annotations

import json
from pathlib import Path
import re
import types

import numpy as np

from room_reconstruction import depth_da3
from room_reconstruction.provenance import sha256_file


def prepare_inputs(camera_run: Path) -> dict:
    """Verify all 52 registered TRAIN images in the accepted 15–45 second pilot."""
    camera_run = Path(camera_run).resolve()
    files = {name: camera_run / f'{name}.json' for name in ('status', 'report', 'manifest')}
    if any(not p.resolve().is_relative_to(camera_run) for p in files.values()):
        raise ValueError('Upstream metadata escapes camera run')
    upstream = {name: json.loads(path.read_text()) for name, path in files.items()}
    if (upstream['status'].get('status') != 'camera_passed'
            or upstream['status'].get('camera_ready') is not True
            or upstream['report'].get('stage') != 'camera_pilot'
            or upstream['manifest'].get('stage') != 'camera_pilot'
            or upstream['report'].get('quality', {}).get('passed') is not True
            or upstream['report'].get('undistorted') is not True):
        raise ValueError('Accepted camera_passed pilot with undistorted inputs required')
    metadata = camera_run / 'undistorted/cameras.json'
    if not metadata.resolve().is_relative_to(camera_run):
        raise ValueError('Camera metadata escapes camera run')
    cameras = json.loads(metadata.read_text())
    if (cameras.get('schema_version') != 1
            or cameras.get('pose_convention') != 'OpenCV c2w; +x right, +y down, +z forward'
            or cameras.get('world') not in ('Relative, unscaled COLMAP world',
                'Relative, unscaled COLMAP world; no gravity or yaw alignment')):
        raise ValueError('Expected schema 1 global relative OpenCV c2w metadata')
    image_root_name = cameras.get('image_root')
    if not isinstance(image_root_name, str) or not image_root_name or Path(image_root_name).is_absolute():
        raise ValueError('Expected relative image_root')
    image_root = (metadata.parent / image_root_name).resolve()
    if not image_root.is_relative_to(metadata.parent.resolve()):
        raise ValueError('Image root escapes undistorted directory')
    train = [row for row in cameras['frames'] if row.get('split') == 'train']
    if len(train) != 52 or any(not np.isfinite(row['timestamp_seconds'])
            or not 15 <= row['timestamp_seconds'] <= 45 for row in train):
        raise ValueError('Pilot requires exactly 52 TRAIN frames within 15–45 seconds')
    frames = sorted(train, key=lambda row: (row['timestamp_seconds'], row['source_index'], row['frame_id']))
    native_rows = upstream['manifest'].get('selected_frames', [])
    native = {row['frame_id']: row for row in native_rows}
    if len(native) != len(native_rows):
        raise ValueError('Duplicate frame IDs in native manifest')
    paths, poses, intrinsics = [], [], []
    for row in frames:
        original = native.get(row['frame_id'])
        if (original is None or original.get('split') != 'train'
                or original.get('source_index') != row['source_index']
                or original.get('timestamp_seconds') != row['timestamp_seconds']):
            raise ValueError('Undistorted frame is not a selected native TRAIN member')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', row['frame_id']):
            raise ValueError('Unsafe frame ID')
        name = row['image_name']
        path = (image_root / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(image_root):
            raise ValueError('Image path escapes image_root')
        if not path.is_file() or sha256_file(path) != row['image_sha256']:
            raise ValueError(f'Image missing or image hash changed: {name}')
        if row['camera_model'] != 'PINHOLE' or row['image_size'] != [1078, 1927]:
            raise ValueError('Expected undistorted PINHOLE images sized 1078×1927')
        K, pose = np.asarray(row['K'], np.float64), np.asarray(row['c2w'], np.float64)
        if (K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0
                or not np.allclose(K[2], [0, 0, 1]) or abs(K[0, 1]) > 1e-8 or abs(K[1, 0]) > 1e-8):
            raise ValueError('Invalid PINHOLE intrinsics')
        if (pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1])
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5)):
            raise ValueError('Invalid proper rigid c2w pose')
        paths.append(str(path))
        intrinsics.append(K)
        poses.append(pose)
    if (len({r['frame_id'] for r in frames}) != 52 or len(set(paths)) != 52
            or len({r['source_index'] for r in frames}) != 52):
        raise ValueError('Duplicate frames/images/source indices')
    poses = np.asarray(poses)
    for window in plan_windows(52):
        centers = poses[window, :3, 3]
        if np.linalg.matrix_rank(centers - centers.mean(axis=0), tol=1e-6) < 2:
            raise ValueError('Window camera centers degenerate for global scale fitting')
    files['cameras'] = metadata
    return dict(frames=frames, image_paths=paths, c2w=poses.astype(np.float32),
        w2c=np.linalg.inv(poses).astype(np.float32), K=np.asarray(intrinsics, np.float32),
        image_size=[1078, 1927], upstream_hashes={str(p): sha256_file(p) for p in files.values()})


def plan_windows(frame_count: int) -> list[list[int]]:
    """Fixed 16-view/stride-12 pilot; reject accidental dataset scope changes."""
    if frame_count != 52:
        raise ValueError('Expected exactly 52 pilot frames')
    return [list(range(start, start + 16)) for start in (0, 12, 24, 36)]


def select_window_frames(windows: list[list[int]]) -> dict[int, dict]:
    """Select a single context by temporal-index center distance, then window order."""
    chosen = {}
    for window_index, indices in enumerate(windows):
        center = (len(indices) - 1) / 2
        for local_index, frame_index in enumerate(indices):
            distance = abs(local_index - center)
            rank = (distance, window_index)
            if frame_index not in chosen or rank < chosen[frame_index]['rank']:
                chosen[frame_index] = dict(window_index=window_index, local_index=local_index,
                    center_distance=distance, rank=rank)
    return dict(sorted(chosen.items()))


def validate_prediction(prediction, K, w2c, image_size) -> dict:
    """Apply the established numeric contract to a complete sixteen-view window."""
    if len(K) != 16 or len(w2c) != 16:
        raise ValueError('Expected sixteen input cameras')
    for name in ('depth', 'conf', 'intrinsics', 'extrinsics', 'processed_images'):
        if len(np.asarray(getattr(prediction, name))) != 16:
            raise ValueError(f'Expected sixteen output {name} views')
    summaries = []
    for start in range(0, 16, 4):
        subset = types.SimpleNamespace(**{name: np.asarray(getattr(prediction, name))[start:start + 4]
            for name in ('depth', 'conf', 'intrinsics', 'extrinsics', 'processed_images')}, is_metric=prediction.is_metric)
        summary = depth_da3.validate_prediction(subset, K[start:start + 4], w2c[start:start + 4], image_size)
        summaries.extend(summary['frames'])
    return dict(summary, shape=list(np.asarray(prediction.depth).shape), frames=summaries)


def attach_alignment_capture(model, api) -> dict:
    """Capture the actual public API fit, including its >=10-view RANSAC policy."""
    captured = {}
    original = model._align_to_input_extrinsics_intrinsics

    def capture(self, extrinsics, intrinsics, prediction, *args, **kwargs):
        captured.clear()
        captured.update({name: np.asarray(getattr(prediction, source), np.float32).copy()
            for name, source in [('extrinsics', 'extrinsics'), ('intrinsics', 'intrinsics'), ('depth', 'depth')]})
        if not np.isfinite(captured['extrinsics']).all() or not np.isfinite(captured['intrinsics']).all():
            raise ValueError('Raw predicted cameras contain nonfinite values')
        align = api.align_poses_umeyama
        calls = []

        def record(*fit_args, **fit_kwargs):
            result = align(*fit_args, **fit_kwargs)
            calls.append(fit_kwargs.copy())
            scale = float(result[2])
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError('Invalid global input-to-prediction scale')
            captured['input_to_prediction_scale'] = scale
            captured['alignment_rotation'] = np.asarray(result[0]).tolist()
            captured['alignment_translation'] = np.asarray(result[1]).tolist()
            return result

        api.align_poses_umeyama = record
        try:
            result = original(extrinsics, intrinsics, prediction, *args, **kwargs)
        finally:
            api.align_poses_umeyama = align
        if len(calls) != 1 or calls[0] != dict(ransac=len(extrinsics) >= 10, return_aligned=True, random_state=42):
            raise ValueError('Installed DA3 alignment API differs from the validated RANSAC contract')
        captured['alignment_arguments'] = calls[0]
        return result

    model._align_to_input_extrinsics_intrinsics = types.MethodType(capture, model)
    return captured


def depth_agreement(a, b) -> dict:
    """Same-camera unfiltered Z comparison; never use these ratios to merge depths."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError('Diagnostic depth shapes differ')
    valid = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    count = int(valid.sum())
    if not count:
        return dict(valid_pair_count=0, valid_pair_fraction=0., median_depth_ratio_b_over_a=None,
            depth_ratio_quantiles=None, agreement_within_10_percent=None, agreement_within_20_percent=None)
    ratio = b[valid].astype(np.float64) / a[valid]
    relative = np.abs(b[valid].astype(np.float64) - a[valid]) / a[valid]
    return dict(valid_pair_count=count, valid_pair_fraction=float(valid.mean()),
        median_depth_ratio_b_over_a=float(np.median(ratio)), depth_ratio_quantiles=np.quantile(ratio, [.05, .5, .95]).tolist(),
        agreement_within_10_percent=float(100 * np.mean(relative <= .1)),
        agreement_within_20_percent=float(100 * np.mean(relative <= .2)))
