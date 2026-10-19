#!/usr/bin/env python3
"""Guarded 16-view DA3-LARGE pilot; preserve global poses and every window output."""
import argparse
from datetime import datetime, timezone
from functools import partial
import importlib.metadata
import json
import os
from pathlib import Path
import re
import resource
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from room_reconstruction.pipeline_layout import resolve_run_dir
from room_reconstruction.provenance import create_run_dir, git_snapshot, load_yaml, sha256_file, write_json
from room_reconstruction.resources import GIB
from smoke_da3 import fresh_resource_guard, progress


def output_run(root, name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', name):
        raise ValueError('Upstream run must be an outputs run ID')
    path = resolve_run_dir(root,name)
    if not path.is_relative_to((root / 'outputs').resolve()):
        raise ValueError('Upstream run escapes outputs')
    return path


def check_budget(torch):
    allocated = torch.cuda.max_memory_allocated(0)
    reserved = torch.cuda.max_memory_reserved(0)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if allocated > 12 * GIB or reserved > 12 * GIB or rss > 6 * GIB:
        raise ValueError('Pilot exceeded allocated/reserved GPU or host RSS budget')
    return dict(torch_peak_allocated_bytes=allocated, torch_peak_reserved_bytes=reserved,
        host_peak_rss_bytes=rss)


def verify_inputs(inputs):
    for path, digest in inputs['upstream_hashes'].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError('Upstream camera artifact changed during pilot')
    for frame, path in zip(inputs['frames'], inputs['image_paths'], strict=True):
        if sha256_file(Path(path)) != frame['image_sha256']:
            raise ValueError('Input image changed during pilot')


def smoke_reference(root, args, inputs, checkpoint, source_hashes):
    """Accept only a successful four-view smoke with matching input and model bytes."""
    run = output_run(root, args.smoke_run)
    files = {name: run / f'{name}.json' for name in ('status', 'manifest', 'report')}
    if any(not p.resolve().is_relative_to(run) for p in files.values()):
        raise ValueError('Smoke metadata escapes its run')
    status, manifest, report = [json.loads(files[name].read_text()) for name in ('status', 'manifest', 'report')]
    if (status.get('status') != 'smoke_passed' or report.get('status') != 'smoke_passed'
            or manifest.get('camera_run') != args.camera_run
            or report.get('checkpoint', {}).get('weights_sha256') != checkpoint['weights_sha256']
            or report.get('vendor_source_hashes') != source_hashes):
        raise ValueError('Comparison requires accepted smoke from the same camera/model/source inputs')
    selected = manifest.get('selected_train_frames', [])
    artifacts = report.get('artifacts', [])
    if len(selected) != 4 or len(artifacts) != 4:
        raise ValueError('Reference smoke must contain exactly four views')
    rows = {row['frame_id']: row for row in inputs['frames']}
    by_id = {artifact['frame_id']: artifact for artifact in artifacts}
    if len(by_id) != 4 or len({row['frame_id'] for row in selected}) != 4:
        raise ValueError('Duplicate smoke frames')
    result = {}
    for row in selected:
        frame_id = row['frame_id']
        if frame_id not in rows or row['image_sha256'] != rows[frame_id]['image_sha256']:
            raise ValueError('Smoke input is not the same pilot TRAIN image')
        artifact = by_id[frame_id]
        path = (run / artifact['file']).resolve()
        if not path.is_relative_to(run) or not path.is_file() or sha256_file(path) != artifact['sha256']:
            raise ValueError('Smoke artifact missing, escaped, or changed')
        result[frame_id] = path
    return result, {str(path): sha256_file(path) for path in files.values()}


def run_pilot(root, run, args):
    started = time.monotonic()
    config = load_yaml(root / args.config)
    if config.get('profile') != 'server' or Path(config['paths']['project_root']).resolve() != root:
        raise ValueError('Pilot requires the declared server project root')
    if Path(sys.prefix).resolve() != (root / '.venv-inference').resolve():
        raise ValueError('Use the isolated .venv-inference/bin/python runtime')
    if 'torch' in sys.modules:
        raise ValueError('Torch imported before the fresh resource and isolation gate')
    fresh_resource_guard(run, config, 'resources_initial.json')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode = True
    from room_reconstruction import depth_pilot as contract
    from room_reconstruction.depth_da3 import validate_snapshot
    import numpy as np
    if np.__version__ != '1.26.4':
        raise ValueError(f'Validated inference runtime requires NumPy 1.26.4; found {np.__version__}')
    inputs = contract.prepare_inputs(output_run(root, args.camera_run))
    windows = contract.plan_windows(len(inputs['frames']))[:args.max_windows]
    selection = contract.select_window_frames(windows)
    from PIL import Image
    for path in inputs['image_paths']:
        with Image.open(path) as image:
            if image.size != tuple(inputs['image_size']):
                raise ValueError('Decoded image dimensions differ from camera metadata')
    registry_path = root / 'configs/models.yaml'
    registry = load_yaml(registry_path)
    if registry.get('schema_version') != 1:
        raise ValueError('Expected schema 1 model registry')
    cache = Path(config['paths']['model_cache'])
    checkpoint = validate_snapshot(cache if cache.is_absolute() else root / cache, registry['models']['da3_large_legacy'])
    progress(run, 'Hashing pinned local DA3-LARGE checkpoint; no inference yet')
    checkpoint['weights_sha256'] = sha256_file(Path(checkpoint['snapshot']) / 'model.safetensors')
    checkpoint['registry_sha256'] = sha256_file(registry_path)
    vendor = root / 'vendor'
    source_files = ['api.py', 'cfg.py', 'specs.py', 'model/da3.py', 'model/cam_enc.py',
        'utils/io/input_processor.py', 'utils/io/output_processor.py', 'utils/pose_align.py']
    source_hashes = {name: sha256_file(vendor / 'depth_anything_3' / name) for name in source_files}
    smoke, smoke_hashes = smoke_reference(root, args, inputs, checkpoint, source_hashes)
    bootstrap = root / 'environments/inference-bootstrap.json'
    window_manifest = [dict(window_index=i, temporal_indices=indices,
        frame_ids=[inputs['frames'][j]['frame_id'] for j in indices]) for i, indices in enumerate(windows)]
    manifest = dict(stage='depth_pilot', created_at_utc=datetime.now(timezone.utc).isoformat(),
        source=git_snapshot(root), arguments=vars(args), camera_run=args.camera_run,
        selected_train_frames=inputs['frames'], upstream_hashes=inputs['upstream_hashes'],
        smoke_run=args.smoke_run, smoke_upstream_hashes=smoke_hashes,
        checkpoint=checkpoint, vendor_source_hashes=source_hashes,
        bootstrap_sha256=sha256_file(bootstrap) if bootstrap.is_file() else None,
        windows=window_manifest, requested_window_count=args.max_windows,
        settings=dict(window_size=16, stride=12, overlap=4, process_res=504,
            process_res_method='upper_bound_resize', align_to_input_ext_scale=True,
            infer_gs=False, use_ray_pose=False, ref_view_strategy='first', export_dir=None,
            cpu_threads=4, torch_interop_threads=1, opencv_threads=1, workers=2,
            torch_memory_cap_bytes=12 * GIB, logical_device='cuda:0'),
        selection='Minimum absolute temporal-index distance to window center; ties choose lower window index',
        interpretation='Unfiltered relative optical-axis Z. Global camera poses retained. Diagnostics do not validate geometry.')
    write_json(run / 'manifest.json', manifest)
    # Fresh telemetry follows hashing/CPU preparation and precedes CUDA import and model loading.
    resources = fresh_resource_guard(run, config, 'resources_preimport.json')
    sys.path.insert(0, str(vendor))
    import torch
    import cv2
    from depth_anything_3 import api
    if Path(api.__file__).resolve() != (vendor / 'depth_anything_3/api.py').resolve():
        raise ValueError('DA3 import must resolve to the workspace vendor source')
    if torch.cuda.device_count() != 1 or os.environ['CUDA_VISIBLE_DEVICES'] != resources['selected_gpu']['uuid']:
        raise ValueError('CUDA must expose exactly the selected physical GPU 0 UUID')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    if cv2.getNumThreads() > 1:
        raise ValueError('OpenCV single-thread cap failed')
    cap_fraction = min(1., 12 * GIB / torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.set_per_process_memory_fraction(cap_fraction, 0)
    torch.cuda.reset_peak_memory_stats(0)
    progress(run, 'Loading offline DA3-LARGE on isolated physical GPU 0')
    model = api.DepthAnything3.from_pretrained(checkpoint['snapshot'], local_files_only=True,
        strict=True).eval().float().to('cuda:0')
    model.input_processor = partial(model.input_processor, num_workers=2)
    raw = contract.attach_alignment_capture(model, api)
    check_budget(torch)
    window_reports, window_artifacts = [], {}
    for window_index, indices in enumerate(windows):
        progress(run, f'Window {window_index + 1}/{len(windows)}: sixteen TRAIN views, 504×280, global camera conditioning')
        with torch.inference_mode():
            prediction = model.inference([inputs['image_paths'][j] for j in indices],
                extrinsics=inputs['w2c'][indices], intrinsics=inputs['K'][indices],
                align_to_input_ext_scale=True, process_res=504, process_res_method='upper_bound_resize',
                infer_gs=False, use_ray_pose=False, ref_view_strategy='first', export_dir=None)
        torch.cuda.synchronize(0)
        numeric = contract.validate_prediction(prediction, inputs['K'][indices], inputs['w2c'][indices], inputs['image_size'])
        if raw['extrinsics'].shape != (16, 3, 4) or raw['intrinsics'].shape != (16, 3, 3):
            raise ValueError('Unexpected raw camera output shape')
        directory = run / f'window_{window_index:03d}'
        directory.mkdir()
        artifacts = []
        for local_index, frame_index in enumerate(indices):
            frame = inputs['frames'][frame_index]
            artifact = directory / f'{frame["frame_id"]}.npz'
            with artifact.open('xb') as stream:
                np.savez_compressed(stream, depth=np.asarray(prediction.depth[local_index], np.float32),
                    conf=np.asarray(prediction.conf[local_index], np.float32),
                    K=np.asarray(prediction.intrinsics[local_index], np.float32),
                    w2c=np.asarray(prediction.extrinsics[local_index], np.float32),
                    processed_rgb=np.asarray(prediction.processed_images[local_index], np.uint8),
                    input_c2w=inputs['c2w'][frame_index], raw_predicted_w2c=raw['extrinsics'][local_index],
                    raw_predicted_K=raw['intrinsics'][local_index], raw_predicted_depth=raw['depth'][local_index])
            row = dict(frame_id=frame['frame_id'], temporal_index=frame_index, local_index=local_index,
                window_index=window_index, file=str(artifact.relative_to(run)),
                sha256=sha256_file(artifact), size_bytes=artifact.stat().st_size)
            artifacts.append(row)
            window_artifacts[(window_index, frame_index)] = row
        window_report = dict(window_manifest[window_index], numeric_contract=numeric, artifacts=artifacts,
            input_to_prediction_trajectory_scale=raw['input_to_prediction_scale'],
            alignment_arguments=raw['alignment_arguments'], alignment_rotation=raw['alignment_rotation'],
            alignment_translation=raw['alignment_translation'], memory=check_budget(torch))
        write_json(directory / 'report.json', window_report)
        window_reports.append(window_report)
        del prediction
        raw.clear()
        torch.cuda.empty_cache()
    verify_inputs(inputs)
    final_artifacts = []
    for frame_index, chosen in selection.items():
        source = window_artifacts[(chosen['window_index'], frame_index)]
        artifact = run / f'{source["frame_id"]}.npz'
        with (run / source['file']).open('rb') as src, artifact.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        final_artifacts.append(dict(source, file=artifact.name, source_window_file=source['file'],
            selection_center_distance=chosen['center_distance'], selection_rule=manifest['selection']))
    overlaps = []
    for window_index in range(1, len(windows)):
        for frame_index in sorted(set(windows[window_index - 1]) & set(windows[window_index])):
            a = window_artifacts[(window_index - 1, frame_index)]
            b = window_artifacts[(window_index, frame_index)]
            with np.load(run / a['file'], allow_pickle=False) as aa, np.load(run / b['file'], allow_pickle=False) as bb:
                diagnostic = contract.depth_agreement(aa['depth'], bb['depth'])
            overlaps.append(dict(frame_id=a['frame_id'], window_a=window_index - 1,
                window_b=window_index, **diagnostic))
    comparisons = []
    final_by_id = {row['frame_id']: row for row in final_artifacts}
    for frame_id, smoke_path in smoke.items():
        if frame_id not in final_by_id:
            continue
        selected = final_by_id[frame_id]
        with np.load(smoke_path, allow_pickle=False) as a, np.load(run / selected['file'], allow_pickle=False) as b:
            if (not np.allclose(a['K'], b['K'], atol=2e-4, rtol=0)
                    or not np.allclose(a['w2c'], b['w2c'], atol=2e-5, rtol=0)
                    or not np.array_equal(a['processed_rgb'], b['processed_rgb'])):
                raise ValueError('Shared smoke input has a different camera or processed RGB contract')
            diagnostic = contract.depth_agreement(a['depth'], b['depth'])
        comparisons.append(dict(frame_id=frame_id, smoke_file=str(smoke_path),
            selected_window_index=selected['window_index'], **diagnostic))
    if len(windows) == 4 and len(comparisons) != 4:
        raise ValueError('Complete pilot must compare all four shared smoke views')
    for path, digest in smoke_hashes.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError('Smoke metadata changed during pilot')
    versions = {}
    for name in ['torch', 'torchvision', 'numpy', 'huggingface-hub', 'safetensors', 'opencv-python-headless', 'Pillow', 'evo']:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = 'not installed under this distribution name'
    complete = len(windows) == 4
    report = dict(stage='depth_pilot', status='depth_passed' if complete else 'resource_proof_passed',
        runtime_validated=True, model_load_validated=True, checkpoint=checkpoint,
        planned_window_count=4, completed_window_count=len(windows), raw_window_frame_count=16 * len(windows),
        input_train_frame_count=52, unique_frame_count=len(final_artifacts), artifacts=final_artifacts,
        windows=window_reports, overlap_depth_diagnostics=overlaps, shared_smoke_comparisons=comparisons,
        diagnostic_definition='b/a ratios and abs(b-a)/a agreement over finite positive pairs, unfiltered; no depth merging',
        depth_alignment_operation='Each window output_depth = raw_depth / actual_global_input_to_prediction_scale',
        camera_frame='Original global relative COLMAP world; no local translation merge',
        predicted_camera_preservation='Raw model E/K saved before API overwrite; conditioned output E/K are not accuracy evidence',
        runtime_versions=versions, vendor_source_hashes=source_hashes,
        gpu_uuid=resources['selected_gpu']['uuid'], gpu_name=torch.cuda.get_device_name(0),
        torch_memory_cap_fraction=cap_fraction, **check_budget(torch),
        cpu_thread_caps=dict(torch=torch.get_num_threads(), torch_interop=torch.get_num_interop_threads(),
            opencv=cv2.getNumThreads(), preprocessing_workers=2), elapsed_seconds=time.monotonic() - started,
        fusion_run=False, scene_optimizer_run=False, geometry_accuracy_validated=False, full_pipeline_ready=False)
    write_json(run / 'report.json', report)
    progress(run, f'{len(windows)} windows / {len(final_artifacts)} unique depth frames passed numeric contracts')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera-run', default='l2bp3_camera_pilot_002')
    parser.add_argument('--smoke-run', default='l2bp3_da3_smoke_001')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--config', default='configs/paths.server.yaml')
    parser.add_argument('--max-windows', type=int, choices=(1, 4), default=4,
        help='One window for the initial resource proof; four for the complete 52-view pilot')
    args = parser.parse_args(argv)
    try:
        run = create_run_dir(ROOT.resolve(), args.run_id)
    except (OSError, ValueError) as exc:
        print(f'Depth pilot: {exc}', file=sys.stderr)
        return 2
    try:
        report = run_pilot(ROOT.resolve(), run, args)
        write_json(run / 'status.json', dict(stage='depth_pilot', status=report['status'],
            depth_ready=report['completed_window_count'] == 4, resource_proof_ready=True,
            completed_window_count=report['completed_window_count'], unique_frame_count=report['unique_frame_count'],
            geometry_accuracy_validated=False, full_pipeline_ready=False))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        failure = dict(stage='depth_pilot', status='failed', reason=str(exc), error_type=type(exc).__name__,
            depth_ready=False, resource_proof_ready=False, full_pipeline_ready=False,
            host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024))
        for filename in ['manifest.json', 'report.json', 'status.json']:
            if not (run / filename).exists():
                write_json(run / filename, dict(failure, arguments=vars(args)))
        print(f'Depth pilot failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
