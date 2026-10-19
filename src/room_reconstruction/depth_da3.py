"""CPU validation for the bounded DA3-LARGE smoke contract; no model imports."""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

from room_reconstruction.provenance import sha256_file


def prepare_inputs(camera_run: Path, start: float = 20, end: float = 26) -> dict:
    """Select exactly four registered training views; verify their undistorted bytes."""
    camera_run = Path(camera_run).resolve()
    if not np.isfinite([start, end]).all() or not 20 <= start < end <= 26:
        raise ValueError('Smoke window must stay within 20–26 seconds')
    files = {name: camera_run / f'{name}.json' for name in ('status', 'report', 'manifest')}
    upstream = {name: json.loads(path.read_text()) for name, path in files.items()}
    if (upstream['status'].get('status') != 'camera_passed'
            or upstream['status'].get('camera_ready') is not True
            or upstream['report'].get('stage') != 'camera_pilot'
            or upstream['manifest'].get('stage') != 'camera_pilot'
            or upstream['report'].get('quality', {}).get('passed') is not True
            or upstream['report'].get('undistorted') is not True):
        raise ValueError('A completed, accepted camera pilot with undistorted inputs is required')
    metadata = camera_run / 'undistorted' / 'cameras.json'
    if not metadata.resolve().is_relative_to(camera_run):
        raise ValueError('Camera metadata escapes its run')
    cameras = json.loads(metadata.read_text())
    if cameras.get('schema_version') != 1 or cameras.get('pose_convention') != 'OpenCV c2w; +x right, +y down, +z forward':
        raise ValueError('Expected schema 1 OpenCV c2w camera metadata')
    image_root = metadata.parent / cameras.get('image_root', '')
    if (not isinstance(cameras.get('image_root'), str)
            or Path(cameras['image_root']).is_absolute()
            or not image_root.resolve().is_relative_to(metadata.parent.resolve())):
        raise ValueError('Image root must remain within the undistorted camera directory')
    candidates = sorted((row for row in cameras['frames'] if row.get('split') == 'train'
        and start <= row['timestamp_seconds'] <= end), key=lambda row: row['timestamp_seconds'])
    if len(candidates) < 4:
        raise ValueError('Smoke requires four registered TRAIN frames in the selected window')
    targets = np.linspace(start, end, 4)
    indices = [min(range(len(candidates)), key=lambda i: abs(candidates[i]['timestamp_seconds'] - t)) for t in targets]
    if len(set(indices)) != 4:
        raise ValueError('Selected time window cannot provide four distinct evenly spaced views')
    frames = [candidates[i] for i in indices]
    native_frames = {row['frame_id']: row for row in upstream['manifest'].get('selected_frames', [])}
    paths, poses, intrinsics = [], [], []
    for row in frames:
        native = native_frames.get(row['frame_id'])
        if (native is None or native.get('split') != 'train'
                or native.get('source_index') != row['source_index']
                or native.get('timestamp_seconds') != row['timestamp_seconds']):
            raise ValueError('Undistorted frame is not a selected native training frame')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', row['frame_id']):
            raise ValueError('Unsafe frame ID')
        name = row['image_name']
        path = image_root / name
        if Path(name).is_absolute() or not path.resolve().is_relative_to(image_root.resolve()):
            raise ValueError('Image path escapes image_root')
        if not path.is_file() or sha256_file(path) != row['image_sha256']:
            raise ValueError(f'Undistorted image missing or hash changed: {name}')
        if row['camera_model'] != 'PINHOLE' or row['image_size'] != [1078, 1927]:
            raise ValueError('Smoke requires undistorted PINHOLE images sized 1078×1927')
        K, pose = np.asarray(row['K'], dtype=np.float64), np.asarray(row['c2w'], dtype=np.float64)
        if (K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0
                or not np.allclose(K[2], [0, 0, 1]) or abs(K[0, 1]) > 1e-8 or abs(K[1, 0]) > 1e-8):
            raise ValueError('Invalid PINHOLE intrinsics')
        if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1])
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5)):
            raise ValueError('Invalid rigid c2w pose')
        paths.append(str(path.resolve()))
        intrinsics.append(K)
        poses.append(pose)
    if len({row['frame_id'] for row in frames}) != 4 or len(set(paths)) != 4:
        raise ValueError('Duplicate frames/images in smoke input')
    poses = np.asarray(poses)
    if np.linalg.matrix_rank(poses[:, :3, 3] - poses[:, :3, 3].mean(axis=0), tol=1e-6) < 2:
        raise ValueError('Camera centers are degenerate for trajectory scale alignment')
    files['cameras'] = metadata
    return dict(frames=frames, image_paths=paths, c2w=poses.astype(np.float32),
        w2c=np.linalg.inv(poses).astype(np.float32), K=np.asarray(intrinsics, dtype=np.float32),
        image_size=[1078, 1927], upstream_hashes={str(p): sha256_file(p) for p in files.values()})


def validate_snapshot(cache: Path, entry: dict) -> dict:
    """Validate pinned local weights by presence/byte size and DA3 architecture config."""
    directory, revision, size = (entry.get(key) for key in ('cache_directory', 'revision', 'weights_bytes'))
    if (entry.get('model_id') != 'depth-anything/DA3-LARGE'
            or directory != 'models--depth-anything--DA3-LARGE'
            or not isinstance(revision, str) or not re.fullmatch('[a-f0-9]{40}', revision)
            or isinstance(size, bool) or not isinstance(size, int) or size <= 0):
        raise ValueError('Expected a pinned DA3-LARGE registry entry')
    snapshot = Path(cache) / directory / 'snapshots' / revision
    config, weights = snapshot / 'config.json', snapshot / 'model.safetensors'
    if not config.is_file() or config.stat().st_size > 1024 * 1024:
        raise ValueError('Model config missing or unexpectedly large')
    if json.loads(config.read_text()).get('model_name') != 'da3-large':
        raise ValueError('Snapshot architecture must be da3-large')
    if not weights.is_file() or weights.stat().st_size != size:
        raise ValueError('Local checkpoint byte size differs from registry')
    return dict(snapshot=str(snapshot), model_id=entry['model_id'], revision=revision,
        weights_bytes=size, config_sha256=sha256_file(config), verification_scope='presence_config_and_byte_size_only')


def expected_resize(K, image_size, process_res=504):
    """Mirror DA3 longest-side resize followed by nearest multiple of fourteen."""
    width, height = image_size
    scale = process_res / max(width, height)
    sizes = [max(1, round(d * scale)) for d in (width, height)]
    rounded = []
    for size in sizes:
        down = size // 14 * 14
        rounded.append(max(1, down + 14 if size - down >= down + 14 - size else down))
    new_width, new_height = rounded
    result = np.asarray(K, dtype=np.float64).copy()
    result[:, 0] *= new_width / width
    result[:, 1] *= new_height / height
    return result.astype(np.float32), (new_height, new_width)


def validate_prediction(prediction, K, w2c, image_size):
    """Check numeric/data contract only; conditioned output cameras do not prove accuracy."""
    expected_K, hw = expected_resize(K, image_size)
    if hw != (504, 280):
        raise ValueError('Unexpected smoke processing resolution')
    depth, conf = np.asarray(prediction.depth), np.asarray(prediction.conf)
    if depth.shape != (4, *hw) or conf.shape != depth.shape or not np.isfinite(conf).all():
        raise ValueError('Invalid depth/confidence shape or nonfinite confidence')
    if (np.asarray(prediction.intrinsics).shape != (4, 3, 3)
            or not np.allclose(prediction.intrinsics, expected_K, rtol=0, atol=2e-4)):
        raise ValueError('Prediction intrinsics differ from exact processed-image resize')
    if (np.asarray(prediction.extrinsics).shape != (4, 3, 4)
            or not np.allclose(prediction.extrinsics, w2c[:, :3], rtol=0, atol=2e-5)):
        raise ValueError('Aligned prediction extrinsics violate the input-pose API contract')
    rgb = np.asarray(prediction.processed_images)
    if rgb.shape != (4, *hw, 3) or rgb.dtype != np.uint8:
        raise ValueError('Invalid processed RGB array')
    if prediction.is_metric:
        raise ValueError('DA3-LARGE smoke must remain relative, nonmetric depth')
    summaries = []
    for d, c in zip(depth, conf, strict=True):
        valid = np.isfinite(d) & (d > 0)
        if not valid.any():
            raise ValueError('A predicted view has no finite positive depth')
        summaries.append(dict(valid_depth_fraction=float(valid.mean()),
            depth_quantiles=np.quantile(d[valid], [0, .05, .5, .95, 1]).tolist(),
            confidence_quantiles=np.quantile(c, [0, .05, .5, .95, 1]).tolist()))
    return dict(shape=list(depth.shape), frames=summaries, depth_definition='OpenCV camera optical-axis Z',
        units='relative COLMAP world units; not meters', is_metric=False,
        camera_contract='API overwrites output E/K with conditioned input E/processed K; this is not an accuracy test',
        geometry_accuracy_validated=False)
