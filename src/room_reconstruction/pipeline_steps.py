"""Server adapters for the established audit, camera, depth and geometry CLIs.

Runs remain immutable. Explicit reuse passes the same identity and artifact gates
as new runs; the small transfer bundle never includes inference checkpoints.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import subprocess

from .provenance import sha256_file
from .pipeline_layout import resolve_run_dir,run_index_path,job_asset


class ComponentDataError(ValueError):
    """A component contract failed; other independent worlds may still pass."""


def _data_call(function, *args):
    try:
        return function(*args)
    except (ValueError, KeyError, TypeError) as exc:
        raise ComponentDataError(str(exc)) from exc


def _component_stage(runner, *args, **kwargs):
    from . import pipeline_state
    try:
        return runner.run(*args, **kwargs)
    except RuntimeError as exc:
        # Resource waits, missing runtimes and unknown backend failures must
        # reach the worker. Only the core's explicit data classification is safe.
        kind = getattr(pipeline_state, 'StageExecutionError', ())
        if isinstance(exc, kind) and exc.quality_failure:
            reason = exc.stage_status.get('reason', str(exc))
            raise ComponentDataError(str(reason)) from exc
        raise


def _json(path):
    return json.loads(Path(path).read_text())


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def _local(root, value, *, outputs=False):
    value = Path(value)
    path = (root / value).resolve()
    base = root / 'outputs' if outputs else root
    if value.is_absolute() or not path.is_relative_to(base.resolve()):
        raise ValueError(f'Workspace-relative path required: {value}')
    return path


def _run(root, name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', name):
        raise ValueError('Safe output run ID required')
    return resolve_run_dir(root,name)


def _status(directory, stage, status):
    state = _json(directory / 'status.json')
    if state.get('stage') != stage or state.get('status') != status:
        raise ValueError(f'{directory.name}: expected {stage}/{status}')


def _artifacts(directory, rows):
    paths = []
    for row in rows:
        value = Path(row['file'])
        path = (directory / value).resolve()
        if value.is_absolute() or not path.is_relative_to(directory.resolve()):
            raise ValueError('Artifact escapes its run')
        if sha256_file(path) != row['sha256']:
            raise ValueError(f'Artifact SHA hash mismatch: {path}')
        if 'size_bytes' in row and path.stat().st_size != row['size_bytes']:
            raise ValueError(f'Artifact size mismatch: {path}')
        paths.append(path)
    return paths


def _output_hashes(root, hashes):
    """Verify recorded generated inputs; code snapshots describe historical code.

    Source files may evolve between runs. Generated camera/depth/surface inputs
    cannot evolve, and absolute server paths must still identify this workspace.
    """
    for name, digest in hashes.items():
        raw = Path(name)
        path = raw.resolve() if raw.is_absolute() else _local(root, name)
        if not path.is_relative_to(root):
            raise ValueError('Recorded provenance path escapes workspace')
        if path.is_relative_to(root / 'outputs') and sha256_file(path) != digest:
            raise ValueError(f'Upstream SHA hash mismatch: {path}')


def camera_matching_mode(processing,frame_count):
    """Bound quadratic matching for short captures; retain explicit overrides."""
    mode=processing.get('matching','auto')
    budget=processing.get('max_exhaustive_frames',400)
    if mode not in ('auto','exhaustive','sequential'):
        raise ValueError('Camera matching must be auto/exhaustive/sequential')
    if type(budget) is not int or budget<=0 or type(frame_count) is not int or frame_count<=0:
        raise ValueError('Camera matching requires positive integer frame count/budget')
    return ('exhaustive' if frame_count<=budget else 'sequential') if mode=='auto' else mode


def capture_config(video, digest, duration, fps=2.0, *, sampling_mode='uniform',
                   adaptive_target_fps=None, motion_threshold=.03):
    duration, fps = float(duration), float(fps)
    if not all(math.isfinite(v) and v > 0 for v in (duration, fps)) or fps > 10:
        raise ValueError('Positive duration and target FPS <= 10 required')
    if sampling_mode not in ('uniform', 'adaptive'):
        raise ValueError('Sampling mode must be uniform or adaptive')
    if adaptive_target_fps is not None:
        adaptive_target_fps = float(adaptive_target_fps)
        if not math.isfinite(adaptive_target_fps) or adaptive_target_fps < fps or adaptive_target_fps > 10:
            raise ValueError('Adaptive target FPS must be finite, >= base FPS and <= 10')
    motion_threshold = float(motion_threshold)
    if not math.isfinite(motion_threshold) or not 0 <= motion_threshold <= 1:
        raise ValueError('Motion threshold must be finite and in [0, 1]')
    return dict(schema_version=1, input_video=str(video),
                verified_metadata=dict(sha256=digest, duration_seconds=duration),
                sampling=dict(target_fps=fps, max_keyframes=math.ceil(duration * (adaptive_target_fps or fps)) + 4,
                              mode=sampling_mode, adaptive_target_fps=adaptive_target_fps,
                              motion_threshold=motion_threshold,
                              selection=('sharpest_per_motion_adaptive_window' if sampling_mode == 'adaptive'
                                         else 'sharpest_per_temporal_window'),
                              preserve_low_texture_continuity=True))


def annotation_config(frames):
    return dict(schema_version=1, manual_review_complete=False,
                review_status='unreviewed; empty manual annotations are not absence evidence',
                frames={f['frame_id']: dict(image_sha256=f['image_sha256'], manual_people_boxes=[],
                                          glass_polygons=[], manual_review_complete=False) for f in frames})


def stage_command(root, stage, **params):
    scripts = dict(audit=('.venv', 'audit_video.py'), camera=('.venv', 'reconstruct_full_cameras.py'),
                   depth=('.venv-inference', 'run_full_depth.py'),
                   surface=('.venv-geometry', 'build_full_surface.py'),
                   render=('.venv-geometry', 'render_full_surface.py'),
                   architecture=('.venv-geometry', 'build_global_architecture.py'),
                   tour=('.venv-geometry', 'export_room_tour.py'))
    runtime, script = scripts[stage]
    command = [str(root / runtime / 'bin/python'), str(root / 'scripts' / script), '--run-id', '@RUN_ID@']
    for name, value in params.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value: command.append('--' + name.replace('_', '-'))
        else:
            command.extend(['--' + name.replace('_', '-'), str(value)])
    return command


def select_components(root, camera, report,job_dir=None):
    from .full_take import prepare_component
    root, camera = Path(root).resolve(), Path(camera).resolve()
    candidates, quarantined = [], []
    for row in report.get('components', []):
        relative = str((camera / row['path']).relative_to(root))
        try:
            if row.get('accepted') is not True:
                raise ValueError('camera quality rejected')
            path = _local(root, relative, outputs=True)
            if not path.is_relative_to(camera): raise ValueError('Component escapes camera run')
            if job_dir is not None:
                from .pipeline_cameras import quarantine_extreme_poses
                path,excluded=quarantine_extreme_poses(root,path,job_dir)
                if excluded:
                    quarantined.append(dict(camera_component=relative,frame_ids=excluded,reason='Extreme poses excluded; valid room views retained'))
                    relative=str(path.relative_to(root))
            inputs = prepare_component(path)
            candidates.append((relative, inputs))
        except (ValueError, KeyError, TypeError) as exc:
            quarantined.append(dict(camera_component=relative, reason=str(exc)))
    candidates.sort(key=lambda item: (-len(item[1]['frames']), item[0]))
    used, selected = set(), []
    for relative, inputs in candidates:
        ids = {f['frame_id'] for f in inputs['frames']}
        # Keep the entire independent world when it contributes any new view.
        # Coverage is the ID union; common views never align these worlds.
        if ids <= used:
            quarantined.append(dict(camera_component=relative, reason='fully redundant accepted frame subset'))
            continue
        index = len(selected)
        selected.append(dict(id='main' if index == 0 else f'component_{index:03d}',
                             name='Area utama' if index == 0 else f'Area terpisah {index}',
                             camera_component=relative, inputs=inputs))
        used.update(ids)
    return selected, quarantined


def _validate_camera(root, directory, audit, digest):
    _status(directory, 'camera_full', 'camera_passed')
    manifest = _json(directory / 'manifest.json')
    if (manifest.get('source_sha256') != digest or manifest.get('audit_run') != audit.name
            or manifest.get('audit_manifest_sha256') != sha256_file(audit / 'frame_manifest.json')):
        raise ValueError('Camera source SHA/audit provenance mismatch')


def _validate_component(root, row, audit, digest):
    from .full_take import prepare_component
    component = _local(root, row['camera_component'], outputs=True)
    inputs = prepare_component(component)
    meta = _json(component / 'manifest.json')
    if meta.get('source_sha256') != digest:
        raise ValueError('Component source SHA mismatch')
    if meta.get('audit_run', audit.name) != audit.name:
        raise ValueError('Component audit provenance mismatch')
    if meta.get('audit_manifest_sha256', sha256_file(audit / 'frame_manifest.json')) != sha256_file(audit / 'frame_manifest.json'):
        raise ValueError('Component audit SHA mismatch')
    audited = {frame['frame_id'] for frame in _json(audit / 'frame_manifest.json')['frames']}
    if not {frame['frame_id'] for frame in inputs['frames']} <= audited:
        raise ValueError('Component contains unaudited frame identities')
    return inputs


def _validate_depth(root, row, inputs):
    directory = _run(root, row['depth_run'])
    _status(directory, 'full_depth', 'depth_passed')
    manifest, report = _json(directory / 'manifest.json'), _json(directory / 'report.json')
    frames = inputs['frames']
    if (manifest.get('camera_component') != row['camera_component'] or manifest.get('selected_frames') != frames
            or report.get('camera_component') != row['camera_component'] or report.get('status') != 'depth_passed'
            or report.get('unique_frame_count') != len(frames)):
        raise ValueError('Depth frame/camera provenance mismatch')
    artifacts = report.get('artifacts', [])
    if (len(artifacts) != len(frames)
            or {r['frame_id'] for r in artifacts} != {f['frame_id'] for f in frames}):
        raise ValueError('Depth must cover each registered frame exactly once')
    hashes = manifest.get('upstream_hashes', {})
    camera = _local(root, row['camera_component'], outputs=True) / 'undistorted/cameras.json'
    recorded = [v for k, v in hashes.items() if k.endswith('/undistorted/cameras.json')]
    if recorded != [sha256_file(camera)]: raise ValueError('Depth camera SHA mismatch')
    _output_hashes(root, hashes)
    _artifacts(directory, artifacts)


def _validate_surface(root, row):
    directory = _run(root, row['surface_run'])
    _status(directory, 'full_surface', 'mesh_passed')
    report, manifest = _json(directory / 'report.json'), _json(directory / 'manifest.json')
    if report.get('depth_run') != row['depth_run'] or report.get('camera_component') != row['camera_component']:
        raise ValueError('Surface depth/camera provenance mismatch')
    _output_hashes(root, manifest.get('input_and_source_hashes', {}))
    for mesh in report.get('meshes', {}).values():
        _artifacts(directory, mesh.get('artifacts', []))


def _validate_render(root, row):
    directory = _run(root, row['render_run'])
    _status(directory, 'full_render', 'render_passed')
    manifest = _json(directory / 'manifest.json')
    if manifest.get('surface_run') != row['surface_run'] or manifest.get('camera_component') != row['camera_component']:
        raise ValueError('Render surface/camera provenance mismatch')
    camera = _local(root, row['camera_component'], outputs=True) / 'undistorted/cameras.json'
    hashes = manifest.get('input_and_source_hashes', {})
    recorded = [v for k, v in hashes.items() if k.endswith('/undistorted/cameras.json')]
    if recorded != [sha256_file(camera)]: raise ValueError('Render camera SHA mismatch')
    _output_hashes(root, hashes)
    files = _artifacts(directory, manifest.get('artifacts', []))
    _artifacts(directory, _json(directory / 'report.json').get('frames', []))
    glbs = [p for p in files if p.suffix == '.glb']
    if len(glbs) != 1: raise ValueError('Exactly one oriented detailed render GLB required')
    return dict(row, mesh=str(glbs[0].relative_to(root)),
                render_manifest=str((directory / 'manifest.json').relative_to(root)),
                cameras=str(camera.relative_to(root)))


def _coverage(audit, selected):
    frames = _json(audit / 'frame_manifest.json')['frames']
    ids = [f['frame_id'] for f in frames]
    if len(set(ids)) != len(ids): raise ValueError('Duplicate audited frame identities')
    processed = set().union(*({f['frame_id'] for f in row['inputs']['frames']} for row in selected))
    if not processed <= set(ids): raise ValueError('Component contains unaudited frames')
    return dict(attempted=len(ids), processed=len(processed), gaps=[fid for fid in ids if fid not in processed],
                full_room_reconstruction_complete=False)


def _tour_config(job, components, coverage):
    return dict(title=job.get('title', Path(job['input_video']).stem + ' · ruang dan lintasan'),
                source=job['input_video'], components=components, coverage=coverage,
                viewer_voxel_divisor=job.get('processing', {}).get('viewer_voxel_divisor', 450),
                limitations=[f"{coverage['processed']} dari {coverage['attempted']} keyframe diproses; "
                             f"{len(coverage['gaps'])} keyframe belum memiliki geometri.",
                             'Komponen memiliki koordinat dan skala terpisah; belum disejajarkan.',
                             'Anotasi manual belum ditinjau. Kaca dan permukaan reflektif dapat menimbulkan artefak.'])


def _validate_tour(root, directory, components):
    _status(directory, 'room_tour_export', 'exported')
    manifest = _json(directory / 'manifest.json')
    hashes = manifest.get('input_hashes', {})
    for row in components:
        for key in ('mesh', 'render_manifest', 'cameras'):
            relative = row[key]
            digest = sha256_file(_local(root, relative, outputs=True))
            matches = (any(v == digest for k, v in hashes.items() if k.endswith('.glb'))
                       if key == 'mesh' else hashes.get(relative) == digest)
            if not matches:
                raise ValueError('Tour input SHA mismatch: ' + relative)
    _output_hashes(root, hashes)
    paths = _artifacts(directory, manifest.get('artifacts', []))
    data = _json(directory / 'tour-data.json')
    if {r['id'] for r in data['components']} != {r['id'] for r in components}:
        raise ValueError('Tour component identity mismatch')
    return paths


def _sample(values, count):
    if len(values) <= count: return values
    return [values[round(i * (len(values) - 1) / (count - 1))] for i in range(count)] if count > 1 else values[:count]


def _bundle_files(root, audit, tour, components, selected, global_component=None):
    tour_components = ([global_component] if global_component else []) + components
    required = {p for p in _validate_tour(root, tour, tour_components) if not p.name.endswith("-sections.json")}
    required.update(tour / name for name in ("status.json", "manifest.json", "tour-data.json"))
    if (tour / "report.json").is_file(): required.add(tour / "report.json")
    required.update(audit / name for name in ("status.json", "frame_manifest.json", "audit_report.json"))
    groups, evidence = [], {}
    for row, internal in zip(components, selected, strict=True):
        required.update(_local(root, row[k], outputs=True) for k in ("mesh", "render_manifest", "cameras"))
        render = _run(root, row["render_run"])
        required.update(render / name for name in ("status.json", "report.json"))
        overview = [p for p in (render / "overview.png", render / "full_path_review.jpg") if p.is_file()]
        source = _sample([Path(p) for p in internal["inputs"]["image_paths"]], 8)
        panels = []
        for frame in _sample(_json(render / "report.json").get("frames", []), 8):
            path = (render / frame["file"]).resolve()
            if not path.is_relative_to(render): raise ValueError("Render QA path escapes run")
            if sha256_file(path) != frame["sha256"]: raise ValueError("Render QA SHA mismatch")
            panels.append(path)
        groups.append((row["id"], dict(source=source, render=panels, overview=overview)))
        evidence[row["id"]] = dict(source=[], render=[], overview=[], complete=False, reason="")
    qa, size = [], 0
    def include(path, component_id=None, category=None):
        nonlocal size
        path = path.resolve()
        if not path.is_relative_to(root / "outputs"): raise ValueError("QA path escapes outputs")
        if not path.is_file(): return False
        if path not in qa:
            if len(qa) >= 32 or size + path.stat().st_size > 20 * 1024**2: return False
            qa.append(path); size += path.stat().st_size
        if component_id is not None:
            relative = str(path.relative_to(root))
            values = evidence[component_id][category]
            if relative not in values: values.append(relative)
        return True
    # First reserve a source image, source/render comparison and overview for
    # every world. Contacts and additional frames cannot displace this minimum.
    for component_id, categories in groups:
        problems = []
        for category, paths in categories.items():
            if not paths: problems.append("missing " + category + " evidence")
            elif not include(paths[0], component_id, category):
                problems.append("QA budget cannot fit required " + category + " evidence")
        record = evidence[component_id]
        record["complete"] = not problems
        record["reason"] = "; ".join(problems)
    # Add remaining component evidence in rounds, so a large main component
    # cannot consume all slots before a later independent component is visited.
    extra_groups = []
    for component_id, categories in groups:
        extras = [(path, component_id, category) for index in range(1, 8)
                  for category, paths in categories.items() for path in paths[index:index + 1]]
        extra_groups.append(extras)
    contacts = []
    for name in _sample(_json(audit / "audit_report.json").get("contact_sheets", []), 6):
        path = (audit / name).resolve()
        if not path.is_relative_to(audit): raise ValueError("Contact sheet escapes audit run")
        contacts.append((path, None, None))
    extra_groups.append(contacts)
    for index in range(max((len(group) for group in extra_groups), default=0)):
        for group in extra_groups:
            if index < len(group): include(*group[index])
    required.update(qa)
    artifacts = [dict(file=str(p.relative_to(root)), sha256=sha256_file(p), size_bytes=p.stat().st_size)
                 for p in sorted(required)]
    return artifacts, [str(p.relative_to(root)) for p in qa], evidence


def _global_stage(root, job_dir, job, components, runner, common):
    config = job_asset(root, job_dir, 'config', 'global-architecture.json')
    inputs = sorted({_local(root, row[key], outputs=True) for row in components
                     for key in ('mesh', 'cameras', 'render_manifest')})
    root_id = ('main' if any(row['id'] == 'main' for row in components) else
               min(components, key=lambda row: (-len(_json(root / row['cameras'])['frames']), row['id']))['id'])
    _write(config, dict(schema_version=1, input_sha256=job['input_sha256'], components=components, root_component=root_id,
        artifacts=[dict(file=str(path.relative_to(root)), sha256=sha256_file(path), size_bytes=path.stat().st_size)
                   for path in inputs]))
    params = dict(config=str(config.relative_to(root)))
    run = runner.run('S06G_global', [config, *inputs], params, stage_command(root, 'architecture', **params),
                     expected=('report.json', 'products/alignment.json', 'products/manifest.json'), **common)
    _status(run, 'global_architecture', 'complete')
    report = _json(run / 'report.json'); products = run / 'products'
    manifest = _json(products / 'manifest.json')
    if manifest.get('input_sha256') != job['input_sha256'] or manifest.get('input_hashes') != {
            str(path.relative_to(root)): sha256_file(path) for path in inputs}:
        raise ValueError('Global architecture input fingerprint differs')
    paths = _artifacts(products, manifest['artifacts'])
    if report != _json(products / 'report.json'):
        # CLI also records the relative products subdirectory.
        if {key: value for key, value in report.items() if key != 'products'} != _json(products / 'report.json'):
            raise ValueError('Global architecture reports differ')
    status, component = report['status'], report['component']
    if status not in ('no_alignment', 'aligned_partial'):
        raise ValueError('Unknown global alignment status')
    if (status == 'no_alignment') != (component is None):
        raise ValueError('Architecture mesh/status contradiction')
    if component:
        if component.get('id') != 'architecture' or len(report['aligned_components']) < 2:
            raise ValueError('Global architecture requires multiple validated worlds')
        for key in ('mesh', 'cameras', 'render_manifest'):
            if _local(root, component[key], outputs=True) not in paths:
                raise ValueError('Architecture component artifact is not hash-bound')
    descriptor = dict(status=status, run_id=run.name, component=component,
        aligned_components=report['aligned_components'], fallback_components=report['fallback_components'],
        alignment=str((products / 'alignment.json').relative_to(root)))
    paths.extend([products / 'manifest.json', run / 'status.json', run / 'report.json'])
    artifacts = [dict(file=str(path.relative_to(root)), sha256=sha256_file(path), size_bytes=path.stat().st_size)
                 for path in sorted(set(paths))]
    return descriptor, artifacts


def run_server(root, job_dir, job, runner):
    root, job_dir = Path(root).resolve(), Path(job_dir).resolve()
    if not job_dir.is_relative_to(root / 'outputs'): raise ValueError('Job must remain within outputs')
    job_dir.mkdir(parents=True, exist_ok=True)
    video = _local(root, job['input_video'])
    if sha256_file(video) != job['input_sha256']: raise ValueError('Input video SHA mismatch')
    reuse, processing = dict(job.get('reuse', {})), job.get('processing', {})
    refinement = job.get('refinement_start')
    if refinement not in (None, 'S06', 'S07'): raise ValueError('Refinement must start at S06 or S07')
    if refinement:
        reuse.pop('tour_run', None)
    if refinement == 'S06':
        reuse['components'] = [{k: v for k, v in row.items() if k not in ('surface_run', 'render_run')}
                               for row in reuse.get('components', [])]
    config = str(_local(root, job['machine_config']).relative_to(root))
    common = dict(timeout=processing.get('stage_timeout_seconds', 21600))
    if reuse.get('audit_run'):
        audit = _run(root, reuse['audit_run'])
    else:
        result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(video)],
                                capture_output=True, text=True, check=True, timeout=60)
        duration = float(json.loads(result.stdout)['format']['duration'])
        capture = job_asset(root,job_dir,'config','capture.json')
        _write(capture, capture_config(
            job['input_video'], job['input_sha256'], duration, processing.get('target_fps', 2),
            sampling_mode=processing.get('sampling_mode', 'uniform'),
            adaptive_target_fps=processing.get('adaptive_target_fps'),
            motion_threshold=processing.get('motion_threshold', .03)))
        params = dict(capture=str(capture.relative_to(root)), threads=4)
        audit = runner.run('S02_audit', [video, capture], params, stage_command(root, 'audit', **params),
                           expected=('frame_manifest.json', 'audit_report.json'), **common)
    _status(audit, 'video_audit', 'complete')
    if _json(audit / 'frame_manifest.json').get('source_sha256') != job['input_sha256']:
        raise ValueError('Audit source SHA mismatch')
    if reuse.get('camera_run'):
        camera = _run(root, reuse['camera_run'])
    else:
        params = dict(audit_run=audit.name, config=config, threads=4,
                      matching=camera_matching_mode(processing,len(_json(audit/'frame_manifest.json')['frames'])),
                      overlap=processing.get('overlap', 16),
                      loop_detection=processing.get('loop_detection', False),
                      vocab_tree_path=processing.get('vocab_tree_path'),
                      max_image_size=processing.get('max_image_size', 960), max_features=processing.get('max_features', 4096))
        camera = runner.run('S03_camera', [audit, root / config], params, stage_command(root, 'camera', **params),
                            expected=('report.json', 'manifest.json'), **common)
    _validate_camera(root, camera, audit, job['input_sha256'])
    # A downstream refinement cannot erase failures of omitted upstream maps.
    failed_components = [dict(item) for item in reuse.get('failed_components',[])]
    def failure(row, stage, exc):
        failed_components.append(dict(id=row.get('id', Path(row.get('camera_component', 'unknown')).name),
            camera_component=row.get('camera_component'), stage=stage, reason=str(exc), error_type=type(exc).__name__))
    if reuse.get('components'):
        selected, quarantined = [], []
        for row in reuse['components']:
            try:
                selected.append(dict(row, inputs=_data_call(_validate_component, root, row, audit, job['input_sha256'])))
            except ComponentDataError as exc:
                failure(row, 'S03_camera', exc)
    else:
        selected, quarantined = select_components(root, camera, _json(camera / 'report.json'),job_dir=job_dir)
        valid = []
        for row in selected:
            try:
                _data_call(_validate_component, root, row, audit, job['input_sha256'])
                valid.append(row)
            except ComponentDataError as exc:
                failure(row, 'S03_camera', exc)
        selected = valid
        for row in quarantined:
            if not row['reason'].startswith('fully redundant'):
                failure(row, 'S03_camera', ComponentDataError(row['reason']))
    if len({r['id'] for r in selected}) != len(selected): raise ValueError('Unique component IDs required')
    if any(not isinstance(row['id'], str) or not re.fullmatch(r'[a-z][a-z0-9_-]{0,39}', row['id'])
           for row in selected):
        raise ValueError('Safe component IDs required')
    _write(job_dir / 'camera-selection.json', dict(selected=[r['camera_component'] for r in selected], quarantined=quarantined))
    components, successful = [], []
    for row in selected:
        stage = 'S05_depth'
        try:
            component = _local(root, row['camera_component'], outputs=True)
            if not row.get('depth_run'):
                params = dict(camera_component=row['camera_component'], config=config)
                depth = _component_stage(runner, 'S05_depth_' + row['id'], [component, root / config, root / 'configs/models.yaml'], params,
                                   stage_command(root, 'depth', **params), gpu=True, expected=('report.json',), **common)
                row['depth_run'] = depth.name
            _data_call(_validate_depth, root, row, row['inputs'])
            stage = 'S06_surface'
            if not row.get('surface_run'):
                annotations = job_asset(root,job_dir,'config','annotations-' + row['id'] + '.json')
                _write(annotations, annotation_config(row['inputs']['frames']))
                params = dict(depth_run=row['depth_run'], camera_component=row['camera_component'],
                              annotations=str(annotations.relative_to(root)),
                              voxel_divisor=processing.get('voxel_divisor', 80), sdf_voxels=processing.get('sdf_voxels', 8),
                              confidence_quantile=processing.get('confidence_quantile', .02))
                surface = _component_stage(runner, 'S06_surface_' + row['id'], [_run(root, row['depth_run']), component, annotations],
                                     params, stage_command(root, 'surface', **params), expected=('report.json',), **common)
                row['surface_run'] = surface.name
            _data_call(_validate_surface, root, row)
            stage = 'S06R_render'
            if not row.get('render_run'):
                params = dict(surface_run=row['surface_run'])
                render = _component_stage(runner, 'S06R_render_' + row['id'], [_run(root, row['surface_run']), component], params,
                                    stage_command(root, 'render', **params), expected=('room-full.glb', 'manifest.json'), **common)
                row['render_run'] = render.name
            public = {k: row[k] for k in ('id', 'name', 'camera_component', 'depth_run', 'surface_run', 'render_run')}
            components.append(_data_call(_validate_render, root, public))
            successful.append(row)
        except ComponentDataError as exc:
            failure(row, stage, exc)
    coverage = _coverage(audit, successful)
    _write(job_dir / 'component-failures.json', dict(failed_components=failed_components, coverage=coverage))
    if not components:
        reasons = '; '.join(f"{row['id']}: {row['reason']}" for row in failed_components)
        raise ComponentDataError('No valid component remains; see component-failures.json. ' + reasons)
    global_info, global_artifacts = None, []
    if processing.get('global_alignment', False):
        global_info, global_artifacts = _global_stage(root, job_dir, job, components, runner, common)
    global_component = global_info['component'] if global_info else None
    if failed_components: reuse.pop('tour_run', None)
    if global_component: reuse.pop('tour_run', None)
    if reuse.get('tour_run'):
        tour = _run(root, reuse['tour_run'])
    else:
        tour_config = job_asset(root,job_dir,'config','tour-config.json')
        tour_components = ([global_component] if global_component else []) + components
        config_data = _tour_config(job, tour_components, coverage)
        if global_info:
            config_data['global_architecture'] = global_info
        _write(tour_config, config_data)
        params = dict(config=str(tour_config.relative_to(root)))
        inputs = [tour_config] + [_local(root, row[k], outputs=True) for row in tour_components for k in ('mesh', 'render_manifest', 'cameras')]
        tour = runner.run('S07_tour', inputs, params, stage_command(root, 'tour', **params),
                          expected=('tour-data.json', 'manifest.json'), **common)
    artifacts, qa, evidence = _bundle_files(root, audit, tour, components, successful, global_component)
    inventory = {item['file']: item for item in artifacts + global_artifacts}
    artifacts = [inventory[key] for key in sorted(inventory)]
    # Resolver metadata is part of the verified transfer, so laptop packaging
    # can locate nested stages without server-specific absolute paths.
    run_ids={audit.name,camera.name,tour.name}
    if global_info: run_ids.add(global_info['run_id'])
    run_ids.update(row[key] for row in components for key in ('depth_run','surface_run','render_run'))
    for identifier in sorted(run_ids):
        index=run_index_path(root,identifier)
        if index.is_file():
            artifacts.append(dict(file=str(index.relative_to(root)),sha256=sha256_file(index),size_bytes=index.stat().st_size))
    bundle = dict(schema_version=1, job_id=job['job_id'], input_sha256=job['input_sha256'],
                  audit_run=audit.name, camera_run=camera.name, coverage=coverage, components=components,
                  tour_run=tour.name, artifacts=artifacts, qa=qa, qa_component_evidence=evidence)
    if failed_components: bundle['failed_components'] = failed_components
    if global_info: bundle['global_architecture'] = global_info
    _write(job_dir / 'bundle.json', bundle)
    return bundle
