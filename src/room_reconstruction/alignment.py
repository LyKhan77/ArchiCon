"""Conservative similarity alignment of independent OpenCV camera worlds.

Shared observations establish a candidate transform, not complete surface
coverage or metric accuracy. Unsupported worlds remain explicitly separate.
"""
from __future__ import annotations

import itertools

import numpy as np

from .camera_geometry import validate_pose


def transform_points(points, matrix):
    points, matrix = np.asarray(points, float), np.asarray(matrix, float)
    if (points.ndim != 2 or points.shape[1] != 3 or matrix.shape != (4, 4)
            or not np.isfinite(points).all() or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1])):
        raise ValueError('Expected finite Nx3 points and a homogeneous 4x4 transform')
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def fit_similarity(source, target):
    """Umeyama source-to-target similarity; reject underdetermined trajectories."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    if (source.ndim != 2 or source.shape[1] != 3 or source.shape != target.shape
            or len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all()):
        raise ValueError('Similarity fitting requires at least three finite paired centers')
    x, y = source - source.mean(0), target - target.mean(0)
    for centered in (x, y):
        singular = np.linalg.svd(centered, compute_uv=False)
        if singular[0] <= 1e-12 or singular[1] / singular[0] < .01:
            raise ValueError('Camera centers are collinear or too poorly conditioned')
    u, singular, vt = np.linalg.svd(y.T @ x / len(x))
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        if singular[-1] > singular[0] * 1e-8:
            raise ValueError('Similarity would require a reflected coordinate frame')
        sign[-1] = -1
    rotation = (u * sign) @ vt
    scale = float(np.dot(singular, sign) / np.mean(np.sum(x * x, axis=1)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Similarity must have a positive finite scale')
    result = np.eye(4)
    result[:3, :3] = scale * rotation
    result[:3, 3] = target.mean(0) - scale * rotation @ source.mean(0)
    return result


def _frames(rows):
    result = {}
    for row in rows:
        identifier = row['frame_id']
        if not isinstance(identifier, str) or not identifier:
            raise ValueError('Nonempty frame IDs required')
        if identifier in result:
            raise ValueError('Duplicate camera frame ID: ' + identifier)
        timestamp = float(row['timestamp_seconds'])
        if not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError('Finite nonnegative camera timestamps required')
        result[identifier] = (timestamp, validate_pose(row['c2w']))
    if not result:
        raise ValueError('Empty camera component')
    return result


def _residuals(source, target, matrix, extent):
    rotation = matrix[:3, :3] / np.cbrt(np.linalg.det(matrix[:3, :3]))
    positions = np.linalg.norm(transform_points(source[:, :3, 3], matrix) - target[:, :3, 3], axis=1) / extent
    expected = rotation @ source[:, :3, :3]
    delta = np.einsum('nji,njk->nik', expected, target[:, :3, :3])
    angles = np.degrees(np.arccos(np.clip((np.trace(delta, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    return dict(center_rms_relative=float(np.sqrt(np.mean(positions ** 2))),
                center_p95_relative=float(np.percentile(positions, 95)),
                center_max_relative=float(positions.max()),
                orientation_max_degrees=float(angles.max()))


def evaluate_edge(source_id, source_rows, target_id, target_rows):
    """Fit on two thirds of shared cameras, validate on disjoint remaining IDs."""
    source, target = _frames(source_rows), _frames(target_rows)
    shared = sorted(source.keys() & target.keys(), key=lambda key: (source[key][0], key))
    result = dict(source=source_id, target=target_id, shared_count=len(shared),
                  shared_frame_ids=shared, accepted=False, reason='', fit_frame_ids=[], validation_frame_ids=[])
    if len(shared) < 6:
        result['reason'] = 'At least six shared cameras required for separate fit and validation'
        return result
    if any(abs(source[key][0] - target[key][0]) > 1e-6 for key in shared):
        raise ValueError('Shared frame timestamps differ across component worlds')
    validation = shared[2::3]; fit = [key for key in shared if key not in set(validation)]
    result.update(fit_frame_ids=fit, validation_frame_ids=validation,
                  thresholds=dict(center_max_relative=.02, orientation_max_degrees=3., min_shared_frames=6))
    a, b = np.array([source[key][1] for key in shared]), np.array([target[key][1] for key in shared])
    extent = float(np.linalg.norm(np.ptp(b[:, :3, 3], axis=0)))
    try:
        matrix = fit_similarity([source[key][1][:3, 3] for key in fit],
                                [target[key][1][:3, 3] for key in fit])
        if extent <= 1e-12:
            raise ValueError('No shared trajectory extent')
        evidence = _residuals(a, b, matrix, extent)
        valid = [shared.index(key) for key in validation]
        result.update(source_to_target=matrix.tolist(), target_extent_relative_units=extent,
                      all_residuals=evidence, validation_residuals=_residuals(a[valid], b[valid], matrix, extent))
        result['accepted'] = evidence['center_max_relative'] <= .02 and evidence['orientation_max_degrees'] <= 3.
        result['reason'] = 'Shared camera validation passed' if result['accepted'] else 'Camera position or orientation residual exceeds threshold'
    except (ValueError, np.linalg.LinAlgError) as exc:
        result['reason'] = str(exc)
    return result


def build_alignment(components, root_id='main'):
    """Maximum-overlap tree connected to root, with non-tree cycle checks."""
    if root_id not in components:
        raise ValueError('Alignment root is missing')
    tables = {key: _frames(rows) for key, rows in sorted(components.items())}
    edges = []
    for a, b in itertools.combinations(sorted(components), 2):
        if tables[a].keys() & tables[b].keys():
            edges.append(evaluate_edge(a, components[a], b, components[b]))
    transforms = {root_id: np.eye(4)}; selected = []
    remaining = sorted((edge for edge in edges if edge['accepted']),
                       key=lambda edge: (-edge['shared_count'], edge['source'], edge['target']))
    while True:
        crossing = [edge for edge in remaining if (edge['source'] in transforms) != (edge['target'] in transforms)]
        if not crossing:
            break
        edge = crossing[0]; a, b = edge['source'], edge['target']
        matrix = np.array(edge['source_to_target'])
        if b in transforms:
            transforms[a] = transforms[b] @ matrix
        else:
            transforms[b] = transforms[a] @ np.linalg.inv(matrix)
        selected.append(dict(source=a, target=b)); remaining.remove(edge)
    cycle_conflicts = []
    for edge in remaining:
        a, b = edge['source'], edge['target']
        if a not in transforms or b not in transforms:
            continue
        keys = edge['shared_frame_ids']
        aa = np.array([tables[a][key][1] for key in keys])
        bb = np.array([tables[b][key][1] for key in keys])
        implied = np.linalg.inv(transforms[b]) @ transforms[a]
        evidence = _residuals(aa, bb, implied, edge['target_extent_relative_units'])
        if evidence['center_max_relative'] > .02 or evidence['orientation_max_degrees'] > 3.:
            cycle_conflicts.append(dict(source=a, target=b, residuals=evidence))
    if cycle_conflicts:
        transforms = {root_id: np.eye(4)}; selected = []
    aligned = [root_id] + sorted(set(transforms) - {root_id})
    return dict(schema_version=1, status='aligned_partial' if len(aligned) > 1 else 'no_alignment',
                root_component=root_id, aligned_components=aligned,
                fallback_components=sorted(set(components) - set(aligned)),
                component_to_root={key: transforms[key].tolist() for key in aligned},
                edges=edges, selected_edges=selected, cycle_conflicts=cycle_conflicts,
                scale_status='relative', full_room_reconstruction_complete=False,
                validation_scope='Shared camera consistency; no independent ground-truth or surface-coverage validation')
