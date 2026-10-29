"""Validated reconstruction scope metadata shared by CLI, dashboard and delivery."""
from __future__ import annotations

import math
from typing import Mapping


KINDS = frozenset({'full_traversal', 'single_room', 'connected_area'})
MAX_NAME_LENGTH = 120


def _number(value, field):
    if isinstance(value, bool):
        raise ValueError(f'{field} must be finite and non-negative')
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field} must be finite and non-negative') from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{field} must be finite and non-negative')
    return value


def normalize_scope(value: Mapping | None, *, duration_seconds: float | None = None) -> dict:
    """Return stable scope metadata; missing scope preserves legacy full traversal."""
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError('scope must be an object')
    kind = value.get('kind', 'full_traversal')
    if kind not in KINDS:
        raise ValueError('scope.kind must be full_traversal, single_room or connected_area')
    name = value.get('name')
    if name is not None:
        if not isinstance(name, str):
            raise ValueError('scope.name must be text')
        name = name.strip()
        if not name or len(name) > MAX_NAME_LENGTH:
            raise ValueError('scope.name must be non-empty and at most 120 characters')
    if kind != 'full_traversal' and name is None:
        raise ValueError('scope.name is required for a scoped reconstruction')
    if duration_seconds is not None:
        duration_seconds = _number(duration_seconds, 'duration_seconds')
    start = value.get('start_time_seconds')
    end = value.get('end_time_seconds')
    start = None if start is None or start == '' else _number(start, 'scope.start_time_seconds')
    end = None if end is None or end == '' else _number(end, 'scope.end_time_seconds')
    if start is not None and end is not None and start >= end:
        raise ValueError('scope.start_time_seconds must be less than end_time_seconds')
    if duration_seconds is not None and end is not None and end > duration_seconds:
        raise ValueError('scope.end_time_seconds exceeds video duration')
    return dict(kind=kind, name=name, start_time_seconds=start,
                end_time_seconds=end,
                source_policy='full_input_video' if start is None and end is None else 'bounded_time_range')
