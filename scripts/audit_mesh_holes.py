#!/usr/bin/env python3
"""CPU triangle visibility audit on fixed held-out views; no hole filling or inference."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import fuse_tsdf as fusion


def checked(root, relative, hashes):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError('Input escapes project')
    hashes[str(path)] = fusion.sha256_file(path)
    return path


def make_views(root, reference, hashes):
    import numpy as np
    path = checked(root, 'outputs/l2bp3_validation_opencv_002/undistorted/cameras.json', hashes)
    status = checked(root, 'outputs/l2bp3_validation_opencv_002/status.json', hashes)
    manifest_path = checked(root, 'outputs/l2bp3_validation_opencv_002/manifest.json', hashes)
    report_path = checked(root, 'outputs/l2bp3_validation_opencv_002/report.json', hashes)
    manifest, report = json.loads(manifest_path.read_text()), json.loads(report_path.read_text())
    if (json.loads(status.read_text()).get('status') != 'localization_passed'
            or manifest.get('stage') != 'validation_localization' or report.get('stage') != 'validation_localization'
            or manifest.get('camera_run') != 'l2bp3_camera_pilot_002'
            or not all(report.get(k) is True for k in ('map_unchanged', 'calibration_fixed', 'training_map_fixed'))
            or any(report.get(k) is not False for k in ('bundle_adjustment_run', 'scene_optimizer_run', 'query_triangulation_run'))
            or report.get('map_hashes_before') != manifest.get('map_hashes_before')
            or report.get('map_hashes_before') != report.get('map_hashes_after')):
        raise ValueError('Accepted fixed-map held-out localization required')
    doc = json.loads(path.read_text())
    calibration = checked(root, 'outputs/l2bp3_camera_pilot_002/undistorted/cameras.json', hashes)
    if (not doc.get('calibration_fixed') or not doc.get('training_map_fixed')
            or doc.get('calibration_reference') != str(calibration)
            or doc.get('calibration_reference_sha256') != hashes[str(calibration)]):
        raise ValueError('Held-out calibration does not match fixed training cameras')
    for file, digest in manifest['map_hashes_before'].items():
        map_path = checked(root, f'outputs/l2bp3_camera_pilot_002/{file}', hashes)
        if hashes[str(map_path)] != digest:
            raise ValueError('Training map changed after held-out localization')
    views = []
    if len(doc['frames']) != 8 or any(row['split'] != 'validation' for row in doc['frames']):
        raise ValueError('Exactly eight held-out views required')
    for row in doc['frames']:
        image = (path.parent / doc['image_root'] / row['image_name']).resolve()
        if not image.is_relative_to(path.parent) or fusion.sha256_file(image) != row['image_sha256']:
            raise ValueError('Held-out image path/hash mismatch')
        hashes[str(image)] = row['image_sha256']
        K = np.asarray(row['K'], dtype=float).copy()
        K[0] *= 280 / row['image_size'][0]
        K[1] *= 504 / row['image_size'][1]
        views.append(dict(id=row['frame_id'], kind='held_out', width=280, height=504,
            K=K.tolist(), c2w=row['c2w'], image=str(image.relative_to(root))))
    transform_path = checked(root, 'outputs/l2bp3_tsdf_001/reports/viewer_export.json', hashes)
    transform = np.asarray(json.loads(transform_path.read_text())['world_to_viewer'])
    bounds = np.asarray(reference['meshes']['filtered']['bounds'])
    corners = np.asarray(np.meshgrid(*zip(bounds[0], bounds[1]))).reshape(3, -1).T
    corners = corners @ transform[:3, :3].T + transform[:3, 3]
    center = (corners.min(0) + corners.max(0)) / 2
    span = float(np.linalg.norm(corners.max(0) - corners.min(0)))
    eye = center + np.asarray([.65, .6, .9]) * span
    forward = center - eye; forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 1, 0]); right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4); pose[:3, :3] = np.column_stack([right, down, forward]); pose[:3, 3] = eye
    pose = np.linalg.inv(transform) @ pose
    views.append(dict(id='exterior', kind='overview', width=960, height=540,
        K=[[800, 0, 480], [0, 800, 270], [0, 0, 1]], c2w=pose.tolist()))
    return views


def boundary_overlay(image, edges, vertices, view, rendered, tolerance):
    """Visible boundary midpoints only; projection is diagnostic, not hole segmentation."""
    import numpy as np
    mid = (vertices[edges[:, 0]] + vertices[edges[:, 1]]) / 2
    E = np.linalg.inv(np.asarray(view['c2w']))
    points = mid @ E[:3, :3].T + E[:3, 3]
    valid = points[:, 2] > 1e-6
    uv = points @ np.asarray(view['K']).T
    uv = uv[valid, :2] / uv[valid, 2:]
    z = points[valid, 2]
    valid = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < view['width'] - .5) & (uv[:, 1] >= 0) & (uv[:, 1] < view['height'] - .5)
    uv = np.rint(uv[valid]).astype(int); z = z[valid]
    scene_z = rendered['depth'][uv[:, 1], uv[:, 0]]
    visible = np.isfinite(scene_z) & (np.abs(scene_z - z) <= tolerance)
    result = image.copy(); uv = uv[visible]
    result[uv[:, 1], uv[:, 0]] = [255, 40, 40]
    return result


def execute(run, args):
    start = time.monotonic()
    if Path(sys.prefix).resolve() != ROOT / '.venv-geometry':
        raise ValueError('Use isolated .venv-geometry')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '4'
    fusion.cpu_guard(run)
    import numpy as np
    import open3d as o3d
    from PIL import Image, ImageDraw
    from room_reconstruction.mesh_audit import TriangleRenderer, pinhole_rays, boundary_edges
    if o3d.__version__ != '0.19.0' or importlib.metadata.version('open3d') != '0.19.0' or np.__version__ != '1.26.4':
        raise ValueError('Validated Open3D 0.19.0 / NumPy 1.26.4 required')
    hashes = {}
    mesh_run = fusion.run_path(ROOT, args.mesh_run)
    for name in ('status', 'manifest', 'report'):
        checked(ROOT, str((mesh_run / f'{name}.json').relative_to(ROOT)), hashes)
    report = json.loads((mesh_run / 'report.json').read_text())
    state = json.loads((mesh_run / 'status.json').read_text())
    mesh_manifest = json.loads((mesh_run / 'manifest.json').read_text())
    if (mesh_manifest.get('camera_run') != 'l2bp3_camera_pilot_002'
            or mesh_manifest.get('stage') != state.get('stage') or report.get('stage') != state.get('stage')):
        raise ValueError('Mesh must use the same accepted global training world as held-out cameras')
    if state.get('status') == 'mesh_passed' and state.get('stage') == 'tsdf_mesh':
        surface = report['meshes']['filtered']
    elif state.get('status') == 'ablation_passed' and state.get('stage') == 'fusion_ablation':
        surface = report['mesh']
    else:
        raise ValueError('Accepted TSDF or ablation mesh required')
    artifact = next(a for a in surface['artifacts'] if a['file'].endswith('.ply'))
    if Path(artifact['file']).is_absolute() or not (mesh_run / artifact['file']).resolve().is_relative_to(mesh_run):
        raise ValueError('Mesh artifact escapes accepted run')
    path = checked(ROOT, str((mesh_run / artifact['file']).relative_to(ROOT)), hashes)
    if hashes[str(path)] != artifact['sha256'] or report['units'] != fusion.UNITS:
        raise ValueError('Mesh provenance or units mismatch')
    reference_path = checked(ROOT, 'outputs/l2bp3_tsdf_001/report.json', hashes)
    reference = json.loads(reference_path.read_text())
    if args.views_run:
        views_path = checked(ROOT, f'outputs/{args.views_run}/views.json', hashes)
        views = json.loads(views_path.read_text())['views']
        # Revalidate fixed cameras, images and reference bounds, not just a caller-supplied view file.
        if views != make_views(ROOT, reference, hashes):
            raise ValueError('Fixed view manifest differs from original held-out cameras')
    else:
        views = make_views(ROOT, reference, hashes)
    for name in ('scripts/audit_mesh_holes.py', 'src/room_reconstruction/mesh_audit.py'):
        checked(ROOT, name, hashes)
    fusion.write_json(run / 'views.json', dict(views=views, reference_mesh_run='l2bp3_tsdf_001',
        overview_up='Estimated camera up; not measured gravity'))
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices = np.asarray(mesh.vertices); triangles = np.asarray(mesh.triangles)
    edges, nonmanifold = boundary_edges(triangles)
    # Boundary topology counted on complete triangle mesh; no vertex sampling.
    from run_fusion_ablation import boundary_diagnostics
    boundary = boundary_diagnostics(vertices, triangles)
    renderer = TriangleRenderer(vertices, triangles, np.asarray(mesh.vertex_colors))
    result = []
    thumbnail_rows = []
    for view in views:
        print(f'Rendering {args.mesh_run}: {view["id"]}', flush=True)
        rays = pinhole_rays(view['K'], view['c2w'], view['width'], view['height'])
        two, one = renderer.render(rays)
        np.savez_compressed(run / f'{view["id"]}.npz', rgb=two['rgb'], depth=two['depth'], hit=two['hit'],
            single_rgb=one['rgb'], single_depth=one['depth'], single_hit=one['hit'])
        def display(render):
            image = np.rint(render['rgb'] * 255).astype(np.uint8)
            image[~render['hit']] = [32, 36, 42]
            return image
        two_image, one_image = display(two), display(one)
        overlay = boundary_overlay(two_image, edges, vertices, view, two, 2 * reference['parameters']['voxel_length'])
        panels = [one_image, two_image, overlay]
        labels = ['Single-sided triangles', 'Double-sided triangles', 'Visible boundary points (red)']
        record = dict(id=view['id'], kind=view['kind'], double_sided_hit_fraction=float(two['hit'].mean()),
            single_sided_hit_fraction=float(one['hit'].mean()),
            backside_only_fraction=float((two['hit'] & ~one['hit']).mean()),
            nearest_backface_fraction=float((two['hit'] & (two['depth'] + 1e-5 < one['depth'])).mean()))
        if 'image' in view:
            source = checked(ROOT, view['image'], hashes)
            target = np.asarray(Image.open(source).convert('RGB').resize((view['width'], view['height']), Image.Resampling.LANCZOS))
            panels.insert(0, target); labels.insert(0, 'Held-out image')
            record['hit_pixel_count'] = int(two['hit'].sum())
            record['rgb_mae_on_hit_0_255'] = (float(np.abs(two['rgb'][two['hit']] * 255 - target[two['hit']]).mean())
                if record['hit_pixel_count'] else None)
            if not record['hit_pixel_count']: record['rgb_error_unavailable_reason'] = 'No ray/triangle hits'
        panel = Image.new('RGB', (view['width'] * len(panels), view['height'] + 28), (245, 245, 245))
        draw = ImageDraw.Draw(panel)
        for i, (array, label) in enumerate(zip(panels, labels)):
            panel.paste(Image.fromarray(array), (i * view['width'], 28)); draw.text((i * view['width'] + 5, 5), label, fill=(0, 0, 0))
        panel.save(run / f'{view["id"]}.png')
        if view['kind'] == 'held_out':
            thumbnail_rows.append(panel.resize((560, 266)))
        result.append(record)
        fusion.check_rss()
    sheet = Image.new('RGB', (1120, 266 * 4))
    for i, image in enumerate(thumbnail_rows): sheet.paste(image, ((i % 2) * 560, (i // 2) * 266))
    sheet.save(run / 'heldout_review.png')
    fusion.verify_hashes(hashes)
    fusion.write_json(run / 'manifest.json', dict(stage='mesh_visibility_audit', mesh_run=args.mesh_run,
        input_and_source_hashes=hashes, source=fusion.git_snapshot(ROOT), actual_triangles=True,
        camera_policy='Fixed held-out PnP cameras; no scene optimization using query images'))
    return dict(stage='mesh_visibility_audit', status='audit_passed', mesh_run=args.mesh_run,
        views=result, boundary=boundary, elapsed_seconds=time.monotonic() - start,
        host_peak_rss_bytes=fusion.check_rss(), units=fusion.UNITS,
        interpretation='Coverage is ray/triangle visibility, not depth accuracy or a true hole count. RGB error includes lighting, pose, masking and color-fusion effects.',
        validation_scope='Eight held-out corridor views in two temporal blocks; not full-room coverage',
        geometry_accuracy_validated=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh-run', required=True); p.add_argument('--run-id', required=True)
    p.add_argument('--views-run')
    args = p.parse_args()
    try: run = fusion.create_run_dir(ROOT, args.run_id)
    except (ValueError, OSError) as exc: print(str(exc), file=sys.stderr); return 2
    try:
        report = execute(run, args)
        fusion.write_json(run / 'report.json', report)
        fusion.write_json(run / 'status.json', dict(stage='mesh_visibility_audit', status='audit_passed'))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        fusion.write_json(run / 'status.json', dict(stage='mesh_visibility_audit', status='failed', reason=str(exc)))
        print(str(exc), file=sys.stderr); return 2


if __name__ == '__main__': raise SystemExit(main())
