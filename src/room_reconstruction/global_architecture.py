"""Immutable global scene candidate built from camera-validated component worlds."""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

from .alignment import build_alignment, transform_points
from .camera_geometry import validate_pose
from .provenance import sha256_file, write_json


def _file(root, name):
    value = Path(name)
    if value.is_absolute() or '..' in value.parts or not value.parts or value.parts[0] != 'outputs':
        raise ValueError('Global architecture input must remain in workspace outputs')
    path = root / value
    if any(parent.is_symlink() for parent in (path, *path.parents) if parent != root.parent):
        raise ValueError('Global architecture input symlink is not allowed')
    if not path.resolve().is_relative_to(root / 'outputs') or not path.is_file():
        raise ValueError('Global architecture input missing or escaped')
    return path


def build_architecture(root, run, config, memory_guard=lambda: None):
    root, run = Path(root).resolve(), Path(run).resolve()
    if not run.is_relative_to(root / 'outputs'):
        raise ValueError('Architecture run must remain in outputs')
    if any(run.iterdir()):
        raise FileExistsError('Architecture run must be empty')
    rows = config['components']; ids = [row['id'] for row in rows]
    if (config.get('schema_version') != 1 or not re.fullmatch(r'[0-9a-f]{64}', config['input_sha256'])
            or len(set(ids)) != len(ids) or not ids or 'architecture' in ids
            or any(not re.fullmatch(r'[a-z][a-z0-9_-]{0,39}', key) for key in ids)):
        raise ValueError('Unique safe components and a video fingerprint required')
    inventory = {item['file']: item for item in config['artifacts']}
    if len(inventory) != len(config['artifacts']):
        raise ValueError('Duplicate architecture input artifacts')
    tables, displays, hashes = {}, {}, {}
    for row in rows:
        for key in ('mesh', 'cameras', 'render_manifest'):
            path = _file(root, row[key]); digest = sha256_file(path)
            item = inventory.get(row[key])
            if item is None or item['sha256'] != digest or item['size_bytes'] != path.stat().st_size:
                raise ValueError('Architecture input checksum or size differs: ' + row[key])
            hashes[row[key]] = digest
        table = json.loads((root / row['cameras']).read_text())
        if table.get('pose_convention') != 'OpenCV c2w; +x right, +y down, +z forward':
            raise ValueError('Architecture requires declared OpenCV c2w cameras')
        tables[row['id']] = table['frames']
        manifest = json.loads((root / row['render_manifest']).read_text())
        displays[row['id']] = validate_pose(manifest['world_to_viewer'])
        if not any(item['sha256'] == hashes[row['mesh']] and Path(item['file']).name == Path(row['mesh']).name
                   for item in manifest['artifacts']):
            raise ValueError('Mesh does not match render provenance')
        recorded = [digest for name, digest in manifest['input_and_source_hashes'].items()
                    if name.endswith('/undistorted/cameras.json')]
        if recorded != [hashes[row['cameras']]]:
            raise ValueError('Cameras do not match render provenance')
    root_id = config.get('root_component', 'main')
    alignment = build_alignment(tables, root_id)
    display = displays[root_id]; component = None; vertex_transforms = {}
    if alignment['status'] != 'no_alignment':
        import trimesh
        from .floorplan import colored_scene_mesh
        meshes, frames = [], {}
        by_id = {row['id']: row for row in rows}
        for identifier in alignment['aligned_components']:
            memory_guard()
            row = by_id[identifier]; similarity = np.asarray(alignment['component_to_root'][identifier])
            # Existing GLBs are in per-component viewer coordinates, not SfM coordinates.
            matrix = display @ similarity @ np.linalg.inv(displays[identifier])
            vertex_transforms[identifier] = matrix.tolist()
            scene = trimesh.load(root / row['mesh'], force='scene', process=False)
            mesh = colored_scene_mesh(scene)
            if (not len(mesh.faces) or not np.isfinite(mesh.vertices).all()
                    or np.asarray(mesh.faces).min() < 0 or np.asarray(mesh.faces).max() >= len(mesh.vertices)):
                raise ValueError('Invalid mesh geometry')
            mesh.apply_transform(matrix); meshes.append(mesh)
            rotation = similarity[:3, :3] / np.cbrt(np.linalg.det(similarity[:3, :3]))
            for frame in tables[identifier]:
                if frame['frame_id'] in frames:
                    continue
                pose = validate_pose(frame['c2w']).copy()
                pose[:3, 3] = transform_points(pose[None, :3, 3], similarity)[0]
                pose[:3, :3] = rotation @ pose[:3, :3]
                frames[frame['frame_id']] = dict(frame, c2w=validate_pose(pose).tolist())
            del scene
        merged = trimesh.util.concatenate(meshes); memory_guard()
        from .mesh_audit import double_sided_glb
        with (run / 'architecture.glb').open('xb') as stream:
            stream.write(double_sided_glb(merged.export(file_type='glb')))
        (run / 'undistorted').mkdir()
        camera = run / 'undistorted/cameras.json'
        write_json(camera, dict(schema_version=1, pose_convention='OpenCV c2w; +x right, +y down, +z forward',
            world=f'Relative root camera world: {root_id}',
            frames=sorted(frames.values(), key=lambda frame: (frame['timestamp_seconds'], frame['frame_id']))))
        # This is merge provenance, not a claim of a fresh image render.
        write_json(run / 'render-manifest.json', dict(stage='global_architecture_merge',
            world_to_viewer=display.tolist(), input_and_source_hashes={str(camera.relative_to(root)): sha256_file(camera)},
            upstream_hashes=hashes, artifacts=[dict(file='architecture.glb', sha256=sha256_file(run / 'architecture.glb'))]))
        component = dict(id='architecture', name='Arsitektur gabungan · kandidat',
                         **{key: str((run / name).relative_to(root)) for key, name in
                            [('mesh', 'architecture.glb'), ('cameras', 'undistorted/cameras.json'),
                             ('render_manifest', 'render-manifest.json')]})
    alignment['mesh_to_root_viewer'] = vertex_transforms
    write_json(run / 'alignment.json', alignment)
    for name, digest in hashes.items():
        if sha256_file(_file(root, name)) != digest:
            raise ValueError('Architecture inputs changed during processing')
    report = dict(stage='global_architecture', status=alignment['status'], component=component,
                  aligned_components=alignment['aligned_components'], fallback_components=alignment['fallback_components'],
                  input_sha256=config['input_sha256'], scale_status='relative',
                  geometry_accuracy_validated=False, full_room_reconstruction_complete=False,
                  fusion_policy='Transform and concatenate observed triangles; no global TSDF, hole filling or duplicate-surface removal')
    write_json(run / 'report.json', report)
    artifacts = [dict(file=str(path.relative_to(run)), sha256=sha256_file(path), size_bytes=path.stat().st_size)
                 for path in sorted(run.rglob('*')) if path.is_file()]
    write_json(run / 'manifest.json', dict(stage='global_architecture', input_hashes=hashes,
        input_sha256=config['input_sha256'], artifacts=artifacts))
    return report
