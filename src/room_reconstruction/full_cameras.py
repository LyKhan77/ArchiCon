"""Production SfM selection and component-local coverage; no held-out quality claims."""
from pathlib import Path

import numpy as np

from .camera_geometry import validate_pose
from .poses import canonical_cameras


def select_full_frames(frames):
    """Use every audited observation while retaining immutable audit split provenance."""
    selected, ids, names = [], set(), set()
    for source in frames:
        row = dict(source)
        name = Path(row['image_path']).name
        if (row['frame_id'] in ids or name in names or row['split'] not in ('train', 'validation')
                or not np.isfinite(row['timestamp_seconds']) or row['timestamp_seconds'] < 0):
            raise ValueError('Full reconstruction requires unique frame IDs/names and valid audited times/splits')
        ids.add(row['frame_id'])
        names.add(name)
        row.update(audit_split=row['split'], reconstruction_role='production')
        selected.append(row)
    if len(selected) < 3:
        raise ValueError('Full reconstruction requires at least three audited frames')
    return sorted(selected, key=lambda row: (row['timestamp_seconds'], row['frame_id']))


def production_cameras(model, frames, component_id):
    cameras = canonical_cameras(model, frames)
    by_id = {row['frame_id']: row for row in frames}
    for row in cameras['frames']:
        row.update(audit_split=by_id[row['frame_id']]['split'], reconstruction_role='production')
    cameras.update(coordinate_frame_id=component_id,
                   world=f'Relative, unscaled component-local COLMAP world: {component_id}; no alignment across components',
                   reconstruction_role='production', production_uses_original_validation=True,
                   evaluation_scope='Reconstruction residuals only; no unbiased held-out evaluation')
    return cameras


def component_quality(errors, cameras):
    rows = cameras['frames']
    calibration_valid = bool(rows)
    try:
        for row in rows:
            k = np.asarray(row['K'], dtype=float)
            size = np.asarray(row['image_size'], dtype=float)
            params = np.asarray(row['camera_params'], dtype=float)
            if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
                    or not np.allclose(k[2], [0, 0, 1]) or size.shape != (2,)
                    or not np.isfinite(size).all() or (size <= 0).any() or not np.isfinite(params).all()):
                raise ValueError('Invalid calibration')
            validate_pose(row['c2w'])
    except (KeyError, TypeError, ValueError):
        calibration_valid = False
    residuals_valid = len(errors) > 0 and np.isfinite(errors).all()
    median, p95 = map(float, np.percentile(errors, [50, 95])) if residuals_valid else (None, None)
    return {'passed': bool(calibration_valid and len(rows) >= 3 and residuals_valid and median <= 1.5 and p95 <= 4),
            'registered_images': len(rows), 'calibration_valid': calibration_valid,
            'observation_count': len(errors), 'median_native_pixels': median, 'p95_native_pixels': p95,
            'mean_native_pixels': float(np.mean(errors)) if residuals_valid else None,
            'thresholds': {'registered_images_min': 3, 'median_native_pixels_max': 1.5, 'p95_native_pixels_max': 4},
            'residual_definition': 'Euclidean per-observation residual using actual camera distortion model',
            'threshold_reference': 'Native source image pixels; no feature-resolution scaling',
            'evaluation_scope': 'Component-local reconstruction fit; no held-out quality claim'}


def coverage_summary(frames, components):
    """Count union coverage and preserve all overlapping model memberships deterministically."""
    selected_ids = {row['frame_id'] for row in frames}
    if any(set(component['frame_ids']) - selected_ids for component in components):
        raise ValueError('Component includes frames outside selected audit scope')
    components = sorted(components, key=lambda row: row['component_id'])
    assignments, missing, unregistered = [], [], []
    accepted_union, registered_union = set(), set()
    for row in frames:
        memberships = [component['component_id'] for component in components if row['frame_id'] in component['frame_ids']]
        accepted = [component['component_id'] for component in components
                    if component['accepted'] and row['frame_id'] in component['frame_ids']]
        primary = (accepted or memberships or [None])[0]
        assignments.append({'frame_id': row['frame_id'], 'timestamp_seconds': row['timestamp_seconds'],
                            'audit_split': row.get('audit_split', row['split']),
                            'component_ids': memberships, 'accepted_component_ids': accepted,
                            'primary_component_id': primary, 'camera_ready': bool(accepted)})
        if memberships:
            registered_union.add(row['frame_id'])
        else:
            unregistered.append(row['frame_id'])
        if accepted:
            accepted_union.add(row['frame_id'])
        else:
            missing.append({'frame_id': row['frame_id'], 'timestamp_seconds': row['timestamp_seconds'],
                            'reason': 'rejected_components_only' if memberships else 'unregistered'})
    complete = len(accepted_union) == len(frames) and bool(frames)
    return {'selected_frame_count': len(frames), 'registered_union_count': len(registered_union),
            'accepted_union_count': len(accepted_union), 'coverage_complete': complete,
            'partial_coverage': not complete, 'frame_assignments': assignments,
            'missing_frames': missing, 'missing_frame_ids': [row['frame_id'] for row in missing],
            'unregistered_frame_ids': unregistered, 'all_selected_frames_accounted': len(assignments) == len(frames),
            'assignment_policy': 'First accepted component, otherwise first returned component, ordered by component_id; all memberships retained',
            'coverage_scope': 'Audited video observations; does not establish coverage of unseen room surfaces'}
