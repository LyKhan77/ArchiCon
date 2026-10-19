"""OpenCV camera axes (+x right, +y down, +z forward); z-depth in scene units."""
import numpy as np


def validate_pose(value):
    pose = np.asarray(value, dtype=float)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-6)):
        raise ValueError("Pose must be a finite 4x4 rigid transform with a proper rotation")
    return pose


def c2w_from_w2c(value):
    pose = np.asarray(value, dtype=float)
    if pose.shape == (3, 4):
        pose = np.vstack([pose, [0, 0, 0, 1]])
    pose = validate_pose(pose)
    result = np.eye(4)
    result[:3, :3] = pose[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ pose[:3, 3]
    return result


def _intrinsics(value):
    k = np.asarray(value, dtype=float)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0
            or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1])):
        raise ValueError("K must be finite 3x3 with positive focal lengths")
    return k


def resize_intrinsics(k, original_size, resized_size):
    """Scale pixel coordinates using (width, height); no half-pixel offset."""
    sizes = np.asarray([original_size, resized_size], dtype=float)
    if sizes.shape != (2, 2) or not np.isfinite(sizes).all() or (sizes <= 0).any():
        raise ValueError("Image dimensions must be positive finite width,height pairs")
    return np.diag([*(sizes[1] / sizes[0]), 1]) @ _intrinsics(k)


def backproject_zdepth(uv, depth, k):
    uv, depth = np.asarray(uv, float), np.asarray(depth, float)
    if (uv.ndim != 2 or uv.shape[1] != 2 or depth.shape != (len(uv),)
            or not np.isfinite(uv).all() or not np.isfinite(depth).all() or (depth <= 0).any()):
        raise ValueError("Expected finite Nx2 pixels and N positive z-depths")
    rays = np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(_intrinsics(k)).T
    return rays * depth[:, None]


def project(xyz, k):
    """Project camera-space points through a pinhole K; distortion handled separately."""
    xyz = np.asarray(xyz, float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all() or (xyz[:, 2] <= 0).any():
        raise ValueError("Expected finite Nx3 points in front of camera")
    pixels = xyz @ _intrinsics(k).T
    return pixels[:, :2] / pixels[:, 2:3]
