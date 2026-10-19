"""Bounded relative-Z filtering; occlusion is absence of evidence, not conflict."""
import numpy as np
from .camera_geometry import validate_pose


def spatial_masks(depth, confidence, semantic_excluded, confidence_quantile=.1, edge_relative=.08):
    d, c, excluded = np.asarray(depth), np.asarray(confidence), np.asarray(semantic_excluded, bool)
    if d.ndim != 2 or c.shape != d.shape or excluded.shape != d.shape:
        raise ValueError('Depth/confidence/semantic masks must share a 2D grid')
    if not 0 <= confidence_quantile < 1 or not np.isfinite(edge_relative) or edge_relative <= 0:
        raise ValueError('Invalid spatial filtering thresholds')
    valid = np.isfinite(d) & (d > 0) & np.isfinite(c)
    eligible = valid & ~excluded
    threshold = float(np.quantile(c[eligible], confidence_quantile)) if eligible.any() else None
    confident = eligible & (c >= threshold) if threshold is not None else np.zeros_like(valid)
    edges = np.zeros_like(valid)
    for axis in (0, 1):
        left, right = [slice(None), slice(None)], [slice(None), slice(None)]
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        a, b = d[left], d[right]
        good = valid[left] & valid[right]
        jump = np.zeros_like(good)
        np.greater(np.abs(a-b), edge_relative*np.minimum(a, b), out=jump, where=good)
        edges[left] |= jump; edges[right] |= jump
    return {'valid': valid, 'semantic': eligible, 'confidence': confident,
            'edge_excluded': edges, 'base': confident & ~edges,
            'confidence_threshold': threshold}


def _frame(frame):
    depth, valid = np.asarray(frame['depth']), np.asarray(frame['valid'], bool)
    K, E = np.asarray(frame['K'], float), np.asarray(frame['w2c'], float)
    if E.shape == (3, 4): E = np.vstack([E, [0, 0, 0, 1]])
    validate_pose(E)
    if (depth.ndim != 2 or valid.shape != depth.shape or K.shape != (3, 3)
            or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0
            or not np.allclose(K[2], [0, 0, 1]) or abs(np.linalg.det(K)) < 1e-10):
        raise ValueError('Invalid depth-grid camera contract')
    return depth, valid & np.isfinite(depth) & (depth > 0), K, E


def view_evidence(source, target, relative_tolerance=.12):
    if not np.isfinite(relative_tolerance) or not 0 < relative_tolerance < 1:
        raise ValueError('Relative tolerance must be in (0,1)')
    d, valid, K, E = _frame(source)
    td, tv, tK, tE = _frame(target)
    yy, xx = np.indices(d.shape)
    pixels = np.stack([xx.ravel(), yy.ravel(), np.ones(d.size)], axis=1)
    safe_depth = np.where(valid, d, 0).ravel()
    xyz = (pixels @ np.linalg.inv(K).T) * safe_depth[:, None]
    # E_target * inverse(E_source): source camera -> world -> target camera.
    transform = tE @ np.linalg.inv(E)
    projected = xyz @ transform[:3, :3].T + transform[:3, 3]
    z = projected[:, 2]
    homogeneous = projected @ tK.T
    positive = valid.ravel() & np.isfinite(projected).all(1) & (z > 0)
    uv = np.zeros((d.size, 2))
    np.divide(homogeneous[:, :2], z[:, None], out=uv, where=positive[:, None])
    # Nearest sample avoids interpolating across foreground depth discontinuities.
    ix, iy = np.rint(uv[:, 0]).astype(int), np.rint(uv[:, 1]).astype(int)
    h, w = td.shape
    inside = (positive & np.isfinite(uv).all(1)
              & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
              & (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h))
    at = np.flatnonzero(inside)
    observable = np.zeros(d.size, bool)
    observable[at] = tv[iy[at], ix[at]]
    at = np.flatnonzero(observable)
    sampled = np.zeros(d.size)
    sampled[at] = td[iy[at], ix[at]]
    delta = z - sampled
    threshold = relative_tolerance * sampled
    support = observable & (np.abs(delta) <= threshold)
    occluded = observable & (delta > threshold)
    contradiction = observable & (delta < -threshold)
    return {name: array.reshape(d.shape) for name, array in {
        'support': support, 'occluded': occluded, 'contradiction': contradiction,
        'unknown': valid.ravel() & ~(support | occluded | contradiction)}.items()}


def multiview_mask(source, neighbors, relative_tolerance=.12, min_support=1):
    if isinstance(min_support, bool) or not isinstance(min_support, int) or min_support < 1:
        raise ValueError('At least one supporting neighbor is required')
    d, valid, _, _ = _frame(source)
    support = np.zeros(d.shape, np.uint16)
    contradictions = np.zeros_like(support); occluded = np.zeros_like(support)
    for target in neighbors:
        evidence = view_evidence(source, target, relative_tolerance)
        support += evidence['support']; contradictions += evidence['contradiction']
        occluded += evidence['occluded']
    # Unsupported surfaces are labelled separately, never called inconsistent.
    return {'mask': valid & (support >= min_support) & (support >= contradictions),
            'support': support, 'contradictions': contradictions, 'occluded': occluded,
            'unsupported': valid & (support == 0)}


def depth_normals(depth, K, mask):
    d = np.asarray(depth); valid = np.asarray(mask, bool) & np.isfinite(d) & (d > 0)
    if d.ndim != 2 or valid.shape != d.shape:
        raise ValueError('Invalid normal depth grid')
    yy, xx = np.indices(d.shape)
    rays = np.stack([xx, yy, np.ones_like(xx)], axis=-1) @ np.linalg.inv(np.asarray(K, float)).T
    points = rays * np.where(valid, d, 0)[..., None]
    dx = points[1:-1, 2:] - points[1:-1, :-2]
    dy = points[2:, 1:-1] - points[:-2, 1:-1]
    values = np.cross(dy, dx)  # Camera-facing normal for a fronto-parallel plane.
    lengths = np.linalg.norm(values, axis=-1)
    inner = (valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2]
             & valid[2:, 1:-1] & valid[:-2, 1:-1] & (lengths > 1e-10))
    normals = np.zeros((*d.shape, 3), np.float32)
    normalized = np.zeros_like(values)
    np.divide(values, lengths[..., None], out=normalized, where=inner[..., None])
    facing = np.sum(normalized * points[1:-1, 1:-1], axis=-1) > 0
    normalized[facing] *= -1
    normals[1:-1, 1:-1] = normalized
    output_valid = np.zeros_like(valid); output_valid[1:-1, 1:-1] = inner
    return normals, output_valid
