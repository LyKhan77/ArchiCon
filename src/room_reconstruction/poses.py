"""Small CPU SfM pilot helpers; validation RGB never enters reconstruction."""
from pathlib import Path
import shutil

import numpy as np

from .camera_geometry import c2w_from_w2c
from .provenance import sha256_file


def select_pilot_frames(frames, start=15, end=45, limit=55):
    if not np.isfinite([start, end]).all() or start < 0 or end <= start or limit < 3:
        raise ValueError("Pilot requires a finite positive interval and at least three frames")
    selected = sorted((row for row in frames if row['split'] == 'train'
                       and start <= row['timestamp_seconds'] < end), key=lambda r: r['timestamp_seconds'])
    if len(selected) < 3:
        raise ValueError("Pilot interval has fewer than three training frames")
    if len(selected) > limit:
        selected = [selected[i] for i in np.linspace(0, len(selected) - 1, limit, dtype=int)]
    return selected


def copy_pilot_images(audit, images, frames):
    audit, images = Path(audit).resolve(), Path(images)
    images.mkdir(exist_ok=True)
    names = set()
    for row in frames:
        source = (audit / row['image_path']).resolve()
        if not source.is_relative_to(audit):
            raise ValueError("Image path escapes audit run")
        if source.name in names:
            raise ValueError("Duplicate image basename")
        names.add(source.name)
        target = images / source.name
        if sha256_file(source) != row['image_sha256']:
            raise ValueError(f"SHA256 mismatch: {source.name}")
        with source.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        if sha256_file(target) != row['image_sha256']:
            raise ValueError(f"Copied SHA256 mismatch: {source.name}")


def observation_residuals(model):
    errors = []
    for image_id in model.reg_image_ids():
        image = model.images[image_id]
        camera = model.cameras[image.camera_id]
        w2c = image.cam_from_world().matrix()
        for observation in image.points2D:
            if observation.has_point3D():
                xyz = model.points3D[observation.point3D_id].xyz
                pixel = camera.img_from_cam(w2c[:, :3] @ xyz + w2c[:, 3])
                if pixel is None or not np.isfinite(pixel).all():
                    raise ValueError("Registered observation has invalid camera projection")
                errors.append(float(np.linalg.norm(pixel - observation.xy)))
    return errors


def quality_summary(errors, registered, selected):
    valid = len(errors) > 0 and np.isfinite(errors).all()
    median, p95 = (map(float, np.percentile(errors, [50, 95])) if valid else (None, None))
    rate = registered / selected if selected else 0
    return {'passed': bool(valid and rate >= .9 and median <= 1.5 and p95 <= 4),
            'registration_rate': rate, 'observation_count': len(errors),
            'median_native_pixels': median, 'p95_native_pixels': p95,
            'mean_native_pixels': float(np.mean(errors)) if valid else None,
            'thresholds': {'registration_rate_min': .9, 'median_native_pixels_max': 1.5,
                           'p95_native_pixels_max': 4},
            'residual_definition': 'Euclidean per-observation residual using actual camera distortion model',
            'threshold_reference': 'Native source image pixels; no feature-resolution scaling'}


def canonical_cameras(model, frames):
    by_name = {Path(row['image_path']).name: row for row in frames}
    result = []
    for image_id in model.reg_image_ids():
        image = model.images[image_id]
        row, camera = by_name[image.name], model.cameras[image.camera_id]
        if [camera.width, camera.height] != row['display_size']:
            raise ValueError("COLMAP camera dimensions differ from native audit image dimensions")
        result.append({'frame_id': row['frame_id'], 'source_index': row['source_index'],
            'timestamp_seconds': row['timestamp_seconds'], 'image_name': image.name,
            'image_sha256': row['image_sha256'], 'split': 'train',
            'camera_id': image.camera_id, 'image_size': [camera.width, camera.height],
            'camera_model': camera.model_name, 'camera_params': camera.params.tolist(),
            'K': camera.calibration_matrix().tolist(),
            'c2w': c2w_from_w2c(image.cam_from_world().matrix()).tolist()})
    return {'schema_version': 1, 'pose_convention': 'OpenCV c2w; +x right, +y down, +z forward',
            'world': 'Relative, unscaled COLMAP world; no gravity or yaw alignment',
            'pixel_units': 'native image pixels', 'frames': sorted(result, key=lambda r: r['timestamp_seconds'])}
