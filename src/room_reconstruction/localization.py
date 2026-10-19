"""Held-out absolute poses against an immutable training map and calibration."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .camera_geometry import c2w_from_w2c
from .provenance import sha256_file


def select_validation_frames(frames, start=15, end=45):
    if not np.isfinite([start, end]).all() or start < 0 or end <= start:
        raise ValueError('Validation requires a finite positive interval')
    selected = sorted((row for row in frames if row['split'] == 'validation'
                       and start <= row['timestamp_seconds'] < end), key=lambda r: r['timestamp_seconds'])
    if len(selected) < 8:
        raise ValueError('Localization requires eight held-out validation frames')
    if len(selected) > 8:
        selected = [selected[i] for i in np.linspace(0, len(selected) - 1, 8, dtype=int)]
    if len({row['image_path'] for row in selected}) != 8:
        raise ValueError('Localization requires eight unique validation images')
    return selected


def validate_upstream(audit_status, camera_status, camera_manifest, audit_run, manifest_sha256):
    if (not isinstance(audit_status, dict) or audit_status.get('stage') != 'video_audit'
            or audit_status.get('status') != 'complete'):
        raise ValueError('Localization requires a completed video_audit')
    if (not isinstance(camera_status, dict) or camera_status.get('status') != 'camera_passed'
            or camera_status.get('camera_ready') is not True):
        raise ValueError('Localization requires camera_passed and camera_ready')
    if (not isinstance(camera_manifest, dict) or camera_manifest.get('audit_run') != audit_run
            or camera_manifest.get('audit_manifest_sha256') != manifest_sha256):
        raise ValueError('Training camera pilot must use this exact audit manifest')


def validate_training_names(names, frames):
    by_name = {Path(row['image_path']).name: row for row in frames}
    if not names or any(name not in by_name or by_name[name]['split'] != 'train' for name in names):
        raise ValueError('Fixed map/database must contain only audited training images')


def map_snapshot(camera_run):
    root = Path(camera_run)
    files = [root / 'database.db', *sorted((root / 'primary').rglob('*'))]
    if not (root / 'primary').is_dir() or not (root / 'database.db').is_file():
        raise ValueError('Training map and database are required')
    return {str(path.relative_to(root)): sha256_file(path) for path in files if path.is_file()}


def resolve_correspondences(keypoints, votes, points3D):
    """One query keypoint per unambiguous map point, voting across train views."""
    keypoints = np.asarray(keypoints, dtype=float)
    if keypoints.ndim != 2 or keypoints.shape[1] < 2 or not np.isfinite(keypoints).all():
        raise ValueError('Invalid query keypoints')
    candidates = defaultdict(Counter)
    for query_index, point_id in votes:
        if not 0 <= query_index < len(keypoints) or point_id not in points3D:
            raise ValueError('Match indexes outside query features or fixed map')
        candidates[int(query_index)][int(point_id)] += 1
    by_point = defaultdict(list)
    for query_index, counts in candidates.items():
        winners = counts.most_common(2)
        if len(winners) == 1 or winners[0][1] > winners[1][1]:
            point_id, count = winners[0]
            by_point[point_id].append((count, query_index))
    indices = []
    for point_id, matches in by_point.items():
        matches.sort(reverse=True)
        if len(matches) == 1 or matches[0][0] > matches[1][0]:
            indices.append((matches[0][1], point_id))
    indices.sort()
    pixels = np.array([keypoints[i, :2] for i, _ in indices], dtype=float).reshape(-1, 2)
    xyz = np.array([points3D[p] for _, p in indices], dtype=float).reshape(-1, 3)
    if not np.isfinite(xyz).all():
        raise ValueError('Fixed map contains non-finite points')
    return pixels, xyz, indices


def pose_residuals(pixels, xyz, camera, cam_from_world):
    pixels, xyz, transform = np.asarray(pixels), np.asarray(xyz), np.asarray(cam_from_world)
    errors = []
    for observed, world in zip(pixels, xyz, strict=True):
        point = transform[:, :3] @ world + transform[:, 3]
        if not np.isfinite(point).all() or point[2] <= 0:
            raise ValueError('Pose inliers must project in front of the camera')
        projected = camera.img_from_cam(point)
        if projected is None or not np.isfinite(projected).all():
            raise ValueError('Invalid distorted camera projection')
        errors.append(float(np.linalg.norm(projected - observed)))
    return errors


def query_quality(errors, correspondence_count):
    valid = len(errors) > 0 and np.isfinite(errors).all()
    median, p95 = map(float, np.percentile(errors, [50, 95])) if valid else (None, None)
    ratio = len(errors) / correspondence_count if correspondence_count else 0
    return {'passed': bool(valid and len(errors) >= 30 and ratio >= .2 and median <= 1.5 and p95 <= 4),
            'inlier_count': len(errors), 'correspondence_count': correspondence_count, 'inlier_ratio': ratio,
            'median_native_pixels': median, 'p95_native_pixels': p95,
            'thresholds': {'inlier_count_min': 30, 'inlier_ratio_min': .2,
                           'median_native_pixels_max': 1.5, 'p95_native_pixels_max': 4}}


def localization_quality(queries):
    accepted = sum(row['passed'] for row in queries)
    return {'passed': len(queries) == 8 and accepted == 8, 'selected_validation_frames': len(queries),
            'localized_validation_frames': accepted, 'required_validation_frames': 8,
            'registration_rate': accepted / len(queries) if queries else 0}


def estimate_query_pose(pixels, xyz, camera, backend):
    result = {'quality': query_quality([], len(pixels)), 'c2w': None}
    if len(pixels) < 30:
        result['reason'] = 'Fewer than 30 unambiguous 2D–3D correspondences'
        return result
    original_params = camera.params.copy()
    estimation = backend.AbsolutePoseEstimationOptions()
    estimation.estimate_focal_length = False
    estimation.ransac.max_error = 4
    estimation.ransac.random_seed = 0
    estimation.ransac.num_threads = 1
    refinement = backend.AbsolutePoseRefinementOptions()
    refinement.refine_focal_length = refinement.refine_extra_params = False
    answer = backend.estimate_and_refine_absolute_pose(pixels, xyz, camera,
        estimation_options=estimation, refinement_options=refinement)
    if not np.array_equal(original_params, camera.params):
        raise ValueError('Absolute pose estimation changed fixed camera calibration')
    if answer is None:
        result['reason'] = 'Absolute pose estimation failed'
        return result
    mask = np.asarray(answer['inlier_mask'], dtype=bool).reshape(-1)
    if mask.shape != (len(pixels),):
        raise ValueError('Pose estimator returned invalid inlier mask')
    w2c = answer['cam_from_world'].matrix()
    errors = pose_residuals(pixels[mask], xyz[mask], camera, w2c)
    result.update(quality=query_quality(errors, len(pixels)), c2w=c2w_from_w2c(w2c).tolist(),
                  inlier_residuals_native_pixels=errors)
    return result


def undistort_queries(run, camera_run, cameras, threads=4):
    """Remap OPENCV queries onto the exact exported training pinhole grid."""
    import cv2
    import json
    from .provenance import write_json
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 4:
        raise ValueError('Query undistortion requires one to four CPU threads')
    cv2.setNumThreads(min(threads, max(1, cv2.getNumThreads())))
    if cv2.getNumThreads() > threads:
        # Some OpenCV backends ignore arbitrary caps; force serial execution.
        cv2.setNumThreads(0)
    if cv2.getNumThreads() > threads:
        raise ValueError('OpenCV backend cannot enforce query undistortion thread limit')
    target_path = Path(camera_run) / 'undistorted/cameras.json'
    if not target_path.is_file():
        return {'available': False, 'reason': 'Training pinhole camera export unavailable; native only'}
    target = json.loads(target_path.read_text())['frames']
    if not target or any(row['camera_model'] != 'PINHOLE' or row['image_size'] != target[0]['image_size']
                         or row['K'] != target[0]['K'] for row in target):
        return {'available': False, 'reason': 'Training pinhole grid is not shared; native only'}
    if any(row['camera_model'] != 'OPENCV' for row in cameras['frames']):
        return {'available': False, 'reason': 'Only OPENCV query remapping supported; native only'}
    output = Path(run) / 'undistorted'
    (output / 'images').mkdir(parents=True)
    pinhole = target[0]
    rows = []
    for frame in cameras['frames']:
        source = Path(run) / 'images' / frame['image_name']
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None or [image.shape[1], image.shape[0]] != frame['image_size']:
            raise ValueError('Query image decode/dimensions mismatch')
        mapx, mapy = cv2.initUndistortRectifyMap(np.array(frame['K']), np.array(frame['camera_params'][4:]),
            None, np.array(pinhole['K']), tuple(pinhole['image_size']), cv2.CV_32FC1)
        rectified = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        ok, encoded = cv2.imencode('.jpg', rectified, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise ValueError('Cannot encode undistorted query image')
        destination = output / 'images' / frame['image_name']
        with destination.open('xb') as stream:
            stream.write(encoded.tobytes())
        rows.append(dict(frame, image_size=pinhole['image_size'], camera_model='PINHOLE',
            camera_params=pinhole['camera_params'], K=pinhole['K'],
            native_image_sha256=frame['image_sha256'], image_sha256=sha256_file(destination)))
    write_json(output / 'cameras.json', dict(cameras, frames=rows, image_root='images',
        pixel_units='undistorted image pixels', calibration_reference=str(target_path),
        calibration_reference_sha256=sha256_file(target_path)))
    return {'available': True, 'image_size': pinhole['image_size'], 'K': pinhole['K'],
            'camera_model': 'PINHOLE', 'opencv_threads': cv2.getNumThreads(),
            'method': 'OpenCV remap onto exact training undistorted grid'}
