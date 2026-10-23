#!/usr/bin/env python3
"""Guarded four-training-view DA3-LARGE smoke; no fusion or scene optimization."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import time
import types
from functools import partial

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from room_reconstruction.pipeline_layout import resolve_run_dir
from room_reconstruction.provenance import create_run_dir, git_snapshot, load_yaml, sha256_file, write_json
from room_reconstruction.resources import (GIB, ResourcePolicy, evaluate_resources,
    parse_compute_process_uuids, parse_gpu_csv, parse_mem_available)


def progress(run, message):
    line = f'{datetime.now(timezone.utc).isoformat()} {message}'
    with (run / 'progress.log').open('a') as stream:
        stream.write(line + '\n')
    print(line, flush=True)


def fresh_resource_guard(run, config, filename):
    """Query current resources without importing a CUDA runtime."""
    compute = config.get('compute', {})
    configured = ResourcePolicy.from_mapping(compute)
    if configured.cpu_threads < 4 or configured.workers < 2:
        raise ValueError('Smoke requires the configured four-thread/two-worker policy')
    policy = ResourcePolicy(min_gpu_free_gib=max(14, configured.min_gpu_free_gib),
        min_host_available_gib=max(8, configured.min_host_available_gib),
        min_disk_free_gib=max(50, configured.min_disk_free_gib), cpu_threads=4, workers=2)
    def query(arguments):
        result = subprocess.run(['nvidia-smi', *arguments, '--format=csv,noheader,nounits'],
            text=True, capture_output=True, timeout=20, check=True)
        return result.stdout
    gpus = parse_gpu_csv(query(['--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu']))
    busy = parse_compute_process_uuids(query(['--query-compute-apps=gpu_uuid']))
    resources = evaluate_resources(gpus=gpus, policy=policy, selector=compute.get('gpu_selector'),
        mem_available_bytes=parse_mem_available(Path('/proc/meminfo').read_text()),
        disk_free_bytes=shutil.disk_usage(run).free, busy_uuids=busy,
        current_visible=os.environ.get('CUDA_VISIBLE_DEVICES'))
    resources['measured_at_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(run / filename, resources)
    if not resources['resource_ready']:
        raise ValueError('; '.join(resources['errors']))
    # The selected physical device may have any host index.  The resource gate
    # pins CUDA_VISIBLE_DEVICES to its UUID, so it is exposed as logical cuda:0
    # inside this process while every other GPU remains hidden.
    os.environ.update(resources['launch_environment'])
    return resources


def run_smoke(root, run, args):
    started = time.monotonic()
    config = load_yaml(root / args.config)
    if config.get('profile') != 'server' or Path(config['paths']['project_root']).resolve() != root:
        raise ValueError('Smoke requires the declared server project root')
    if Path(sys.prefix).resolve() != (root / '.venv-inference').resolve():
        raise ValueError('Use the isolated .venv-inference/bin/python runtime')
    if 'torch' in sys.modules:
        raise ValueError('Torch was imported before the fresh resource and isolation gate')
    resources = fresh_resource_guard(run, config, 'resources_initial.json')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode = True
    from room_reconstruction import depth_da3 as contract
    import numpy as np
    if np.__version__ != '1.26.4':
        raise ValueError(f'Validated inference runtime requires NumPy 1.26.4; found {np.__version__}')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', args.camera_run):
        raise ValueError('Camera run must be an outputs run ID')
    camera_run = resolve_run_dir(root,args.camera_run)
    if not camera_run.is_relative_to((root / 'outputs').resolve()):
        raise ValueError('Camera run escapes workspace outputs')
    inputs = contract.prepare_inputs(camera_run)
    from PIL import Image
    for path in inputs['image_paths']:
        with Image.open(path) as image:
            if image.size != tuple(inputs['image_size']):
                raise ValueError('Decoded input dimensions differ from camera metadata')
    registry_path = root / 'configs/models.yaml'
    registry = load_yaml(registry_path)
    if registry.get('schema_version') != 1:
        raise ValueError('Expected schema 1 model registry')
    cache = Path(config['paths']['model_cache'])
    cache = cache if cache.is_absolute() else root / cache
    checkpoint = contract.validate_snapshot(cache, registry['models']['da3_large_legacy'])
    progress(run, 'Hashing pinned local DA3-LARGE checkpoint; no inference yet')
    checkpoint['weights_sha256'] = sha256_file(Path(checkpoint['snapshot']) / 'model.safetensors')
    checkpoint['registry_sha256'] = sha256_file(registry_path)
    vendor = root / 'vendor'
    source_files = ['api.py', 'cfg.py', 'specs.py', 'model/da3.py', 'model/cam_enc.py',
        'utils/io/input_processor.py', 'utils/io/output_processor.py', 'utils/pose_align.py']
    source_hashes = {name: sha256_file(vendor / 'depth_anything_3' / name) for name in source_files}
    bootstrap = root / 'environments/inference-bootstrap.json'
    manifest = dict(stage='da3_smoke', created_at_utc=datetime.now(timezone.utc).isoformat(),
        source=git_snapshot(root), camera_run=args.camera_run, arguments=vars(args),
        selected_train_frames=inputs['frames'], upstream_hashes=inputs['upstream_hashes'],
        checkpoint=checkpoint, vendor_source_hashes=source_hashes,
        bootstrap_sha256=sha256_file(bootstrap) if bootstrap.is_file() else None,
        settings=dict(process_res=504, process_res_method='upper_bound_resize',
            align_to_input_ext_scale=True, infer_gs=False, use_ray_pose=False,
            ref_view_strategy='first', export_dir=None, cpu_threads=4, workers=2,
            torch_memory_cap_bytes=12 * GIB, logical_device='cuda:0'),
        interpretation='Soft camera conditioning; relative Z depth. Four-view smoke does not validate geometry or full pipeline.')
    write_json(run / 'manifest.json', manifest)
    # Hashing/CPU preparation can take time: recheck immediately before runtime import.
    resources = fresh_resource_guard(run, config, 'resources_preimport.json')
    sys.path.insert(0, str(vendor))
    import torch
    import cv2
    from depth_anything_3 import api
    if Path(api.__file__).resolve() != (vendor / 'depth_anything_3/api.py').resolve():
        raise ValueError('DA3 import must resolve to this workspace vendor source')
    if torch.cuda.device_count() != 1 or os.environ['CUDA_VISIBLE_DEVICES'] != resources['selected_gpu']['uuid']:
        raise ValueError('CUDA isolation must expose exactly the selected physical UUID')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    if cv2.getNumThreads() > 1:
        raise ValueError('OpenCV failed to honor its single-thread preprocessing cap')
    cap_fraction = min(1.0, 12 * GIB / torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.set_per_process_memory_fraction(cap_fraction, 0)
    torch.cuda.reset_peak_memory_stats(0)
    progress(run, 'Loading DA3-LARGE from verified local snapshot on isolated cuda:0')
    model = api.DepthAnything3.from_pretrained(checkpoint['snapshot'], local_files_only=True,
        strict=True).eval().float().to('cuda:0')
    # The public API defaults to eight preprocessing workers; bind the bounded policy.
    model.input_processor = partial(model.input_processor, num_workers=2)
    raw = {}
    original_alignment = model._align_to_input_extrinsics_intrinsics
    def capture_alignment(self, extrinsics, intrinsics, prediction, *alignment_args, **kwargs):
        raw['extrinsics'] = np.asarray(prediction.extrinsics, dtype=np.float32).copy()
        raw['intrinsics'] = np.asarray(prediction.intrinsics, dtype=np.float32).copy()
        _, _, scale = api.align_poses_umeyama(raw['extrinsics'], extrinsics.numpy(),
            ransac=False, random_state=42)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Invalid input-to-prediction trajectory scale')
        raw['input_to_prediction_scale'] = float(scale)
        return original_alignment(extrinsics, intrinsics, prediction, *alignment_args, **kwargs)
    model._align_to_input_extrinsics_intrinsics = types.MethodType(capture_alignment, model)
    progress(run, 'Running exactly four training views at 504×280; Gaussian branch disabled')
    prediction = model.inference(inputs['image_paths'], extrinsics=inputs['w2c'], intrinsics=inputs['K'],
        align_to_input_ext_scale=True, process_res=504, process_res_method='upper_bound_resize',
        infer_gs=False, use_ray_pose=False, ref_view_strategy='first', export_dir=None)
    torch.cuda.synchronize(0)
    summary = contract.validate_prediction(prediction, inputs['K'], inputs['w2c'], inputs['image_size'])
    if not np.isfinite(raw['extrinsics']).all() or not np.isfinite(raw['intrinsics']).all():
        raise ValueError('Raw predicted cameras contain nonfinite values')
    for path, digest in inputs['upstream_hashes'].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError('Upstream camera artifacts changed during smoke')
    for frame, path in zip(inputs['frames'], inputs['image_paths'], strict=True):
        if sha256_file(Path(path)) != frame['image_sha256']:
            raise ValueError('Input image bytes changed during smoke')
    artifacts = []
    for i, frame in enumerate(inputs['frames']):
        artifact = run / f'{frame["frame_id"]}.npz'
        with artifact.open('xb') as stream:
            np.savez_compressed(stream, depth=np.asarray(prediction.depth[i], np.float32),
                conf=np.asarray(prediction.conf[i], np.float32),
                K=np.asarray(prediction.intrinsics[i], np.float32),
                w2c=np.asarray(prediction.extrinsics[i], np.float32),
                processed_rgb=np.asarray(prediction.processed_images[i], np.uint8),
                input_c2w=inputs['c2w'][i], raw_predicted_w2c=raw['extrinsics'][i],
                raw_predicted_K=raw['intrinsics'][i])
        artifacts.append(dict(frame_id=frame['frame_id'], file=artifact.name,
            sha256=sha256_file(artifact), size_bytes=artifact.stat().st_size))
    allocated, reserved = torch.cuda.max_memory_allocated(0), torch.cuda.max_memory_reserved(0)
    rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux runtime required.
    if allocated > 12 * GIB or reserved > 12 * GIB or rss_bytes > 6 * GIB:
        raise ValueError('Smoke exceeded the allocated/reserved GPU or host RSS budget')
    versions = {}
    for name in ['torch', 'torchvision', 'numpy', 'huggingface-hub', 'safetensors', 'opencv-python-headless', 'Pillow', 'evo']:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = 'not installed under this distribution name'
    report = dict(stage='da3_smoke', status='smoke_passed', runtime_validated=True, model_load_validated=True,
        numeric_contract=summary, checkpoint=checkpoint, artifacts=artifacts,
        input_to_prediction_trajectory_scale=raw['input_to_prediction_scale'],
        depth_alignment_operation='output_depth = raw_depth / input_to_prediction_trajectory_scale',
        predicted_camera_preservation='Raw model E/K saved separately before public API overwrite; conditioned output E/K are not accuracy evidence',
        runtime_versions=versions, vendor_source_hashes=source_hashes,
        gpu_uuid=resources['selected_gpu']['uuid'], gpu_name=torch.cuda.get_device_name(0),
        torch_memory_cap_fraction=cap_fraction, torch_peak_allocated_bytes=allocated,
        torch_peak_reserved_bytes=reserved, host_peak_rss_bytes=rss_bytes,
        cpu_thread_caps=dict(torch=torch.get_num_threads(), torch_interop=torch.get_num_interop_threads(),
            opencv=cv2.getNumThreads(), preprocessing_workers=2),
        elapsed_seconds=time.monotonic() - started, fusion_run=False, scene_optimizer_run=False,
        geometry_accuracy_validated=False, full_pipeline_ready=False)
    write_json(run / 'report.json', report)
    progress(run, 'Four-view numeric smoke passed; geometry accuracy remains unvalidated')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera-run', default='l2bp3_camera_pilot_002')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--config', default='configs/paths.server.yaml')
    args = parser.parse_args(argv)
    try:
        run = create_run_dir(ROOT.resolve(), args.run_id)
    except (OSError, ValueError) as exc:
        print(f'DA3 smoke: {exc}', file=sys.stderr)
        return 2
    try:
        run_smoke(ROOT.resolve(), run, args)
        write_json(run / 'status.json', dict(stage='da3_smoke', status='smoke_passed',
            bounded_smoke_ready=True, inference_ready=False, geometry_accuracy_validated=False))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        failure = dict(stage='da3_smoke', status='failed', reason=str(exc),
            error_type=type(exc).__name__, bounded_smoke_ready=False, inference_ready=False)
        failure['host_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        for filename in ['manifest.json', 'report.json', 'status.json']:
            if not (run / filename).exists():
                write_json(run / filename, dict(failure, arguments=vars(args)))
        print(f'DA3 smoke failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
