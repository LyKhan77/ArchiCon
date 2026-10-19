#!/usr/bin/env python3
"""Bounded cached DA3-LARGE inference for every registered full-video component view."""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from room_reconstruction.provenance import create_run_dir,write_json,sha256_file,git_snapshot,load_yaml
from smoke_da3 import fresh_resource_guard,progress
from run_depth_pilot import check_budget,verify_inputs


def execute(run,args):
    start=time.monotonic();root=ROOT.resolve()
    if Path(sys.prefix).resolve()!=root/'.venv-inference':raise ValueError('Use isolated inference runtime')
    config=load_yaml(root/getattr(args,'config','configs/paths.server.yaml'))
    if Path(config['paths']['project_root']).resolve()!=root:raise ValueError('Server root required')
    if 'torch' in sys.modules:raise ValueError('CUDA imported before resource gate')
    fresh_resource_guard(run,config,'resources_initial.json')
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='4'
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode=True
    import numpy as np
    from PIL import Image
    from room_reconstruction.full_take import prepare_component,plan_windows,validate_prediction,UNITS
    from room_reconstruction.depth_pilot import select_window_frames,attach_alignment_capture,depth_agreement
    from room_reconstruction.depth_da3 import validate_snapshot
    if np.__version__!='1.26.4':raise ValueError('Validated inference NumPy1.26.4 required')
    component=(root/args.camera_component).resolve()
    if Path(args.camera_component).is_absolute() or not component.is_relative_to(root/'outputs'):
        raise ValueError('Camera component must remain within project outputs')
    inputs=prepare_component(component);planned=plan_windows(len(inputs['frames']))
    windows=planned[:args.max_windows] if args.max_windows else planned
    selected=select_window_frames(windows)
    for path in inputs['image_paths']:
        with Image.open(path) as image:
            if list(image.size)!=inputs['image_size']:raise ValueError('Decoded image size differs from cameras')
    registry_path=root/'configs/models.yaml';registry=load_yaml(registry_path)
    cache=Path(config['paths']['model_cache']);cache=cache if cache.is_absolute() else root/cache
    checkpoint=validate_snapshot(cache,registry['models']['da3_large_legacy'])
    checkpoint['weights_sha256']=sha256_file(Path(checkpoint['snapshot'])/'model.safetensors')
    checkpoint['registry_sha256']=sha256_file(registry_path)
    vendor=root/'vendor'
    source_names=['api.py','cfg.py','specs.py','model/da3.py','model/cam_enc.py',
        'utils/io/input_processor.py','utils/io/output_processor.py','utils/pose_align.py']
    source_hashes={name:sha256_file(vendor/'depth_anything_3'/name) for name in source_names}
    manifest=dict(stage='full_depth',source=git_snapshot(root),arguments=vars(args),
        camera_component=str(component.relative_to(root)),selected_frames=inputs['frames'],
        upstream_hashes=inputs['upstream_hashes'],checkpoint=checkpoint,vendor_source_hashes=source_hashes,
        planned_windows=planned,executed_windows=windows,production_uses_original_validation=True,
        settings=dict(window_size_max=16,stride=12,process_res=504,align_to_input_ext_scale=True,
            infer_gs=False,ref_view_strategy='first',cpu_threads=4,workers=2,torch_allocator_gib=12),
        units=UNITS,world_scope=str(component.relative_to(root)),
        evaluation='Production reconstruction, not held-out evaluation')
    write_json(run/'manifest.json',manifest)
    resources=fresh_resource_guard(run,config,'resources_preimport.json')
    sys.path.insert(0,str(vendor))
    import torch
    import cv2
    from depth_anything_3 import api
    if Path(api.__file__).resolve()!=vendor/'depth_anything_3/api.py':raise ValueError('Unexpected DA3 import source')
    if torch.cuda.device_count()!=1 or os.environ['CUDA_VISIBLE_DEVICES']!=resources['selected_gpu']['uuid']:
        raise ValueError('Expose only selected physical GPU0 UUID')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    torch.cuda.set_per_process_memory_fraction(min(1.,12*1024**3/torch.cuda.get_device_properties(0).total_memory),0)
    torch.cuda.reset_peak_memory_stats(0)
    progress(run,f'Loading local DA3-LARGE: {len(inputs["frames"])} registered production views, {len(windows)} windows')
    model=api.DepthAnything3.from_pretrained(checkpoint['snapshot'],local_files_only=True,strict=True).eval().float().to('cuda:0')
    model.input_processor=partial(model.input_processor,num_workers=2)
    raw=attach_alignment_capture(model,api);check_budget(torch)
    records={};reports=[]
    for wi,indices in enumerate(windows):
        progress(run,f'Window {wi+1}/{len(windows)}: {len(indices)} views in fixed component world')
        with torch.inference_mode():
            prediction=model.inference([inputs['image_paths'][j] for j in indices],
                extrinsics=inputs['w2c'][indices],intrinsics=inputs['K'][indices],align_to_input_ext_scale=True,
                process_res=504,process_res_method='upper_bound_resize',infer_gs=False,use_ray_pose=False,
                ref_view_strategy='first',export_dir=None)
        torch.cuda.synchronize(0)
        numeric=validate_prediction(prediction,inputs['K'][indices],inputs['w2c'][indices],inputs['image_size'])
        scale=raw['input_to_prediction_scale']
        if not np.allclose(np.asarray(prediction.depth),raw['depth']/scale,atol=1e-5,rtol=1e-6):
            raise ValueError('Aligned depth differs from captured actual trajectory scale')
        directory=run/f'window_{wi:03d}';directory.mkdir();artifacts=[]
        for li,fi in enumerate(indices):
            frame=inputs['frames'][fi];path=directory/f'{frame["frame_id"]}.npz'
            with path.open('xb') as stream:
                np.savez_compressed(stream,depth=np.asarray(prediction.depth[li],np.float32),
                    conf=np.asarray(prediction.conf[li],np.float32),K=np.asarray(prediction.intrinsics[li],np.float32),
                    w2c=np.asarray(prediction.extrinsics[li],np.float32),processed_rgb=np.asarray(prediction.processed_images[li],np.uint8),
                    input_c2w=inputs['c2w'][fi],raw_predicted_w2c=raw['extrinsics'][li],
                    raw_predicted_K=raw['intrinsics'][li],raw_predicted_depth=raw['depth'][li])
            artifact=dict(frame_id=frame['frame_id'],timestamp_seconds=frame['timestamp_seconds'],
                window_index=wi,temporal_index=fi,local_index=li,file=str(path.relative_to(run)),
                sha256=sha256_file(path),size_bytes=path.stat().st_size)
            artifacts.append(artifact);records[(wi,fi)]=artifact
        centers=inputs['c2w'][indices,:3,3]
        window_report=dict(window_index=wi,temporal_indices=indices,artifacts=artifacts,numeric_contract=numeric,
            alignment_scale=scale,alignment_arguments=raw['alignment_arguments'],
            alignment_rotation=raw['alignment_rotation'],alignment_translation=raw['alignment_translation'],
            camera_center_rank=int(np.linalg.matrix_rank(centers-centers.mean(0),tol=1e-6)),memory=check_budget(torch))
        write_json(directory/'report.json',window_report);reports.append(window_report)
        del prediction;raw.clear();torch.cuda.empty_cache()
    artifacts=[]
    for fi,choice in selected.items():
        source=records[(choice['window_index'],fi)];path=run/f'{source["frame_id"]}.npz'
        with (run/source['file']).open('rb') as src,path.open('xb') as dst:shutil.copyfileobj(src,dst)
        artifacts.append(dict(source,file=path.name,source_window_file=source['file'],
            selection_center_distance=choice['center_distance']))
    overlaps=[]
    for wi in range(1,len(windows)):
        for fi in sorted(set(windows[wi-1])&set(windows[wi])):
            with np.load(run/records[(wi-1,fi)]['file']) as a,np.load(run/records[(wi,fi)]['file']) as b:
                overlaps.append(dict(frame_id=inputs['frames'][fi]['frame_id'],window_a=wi-1,window_b=wi,
                    **depth_agreement(a['depth'],b['depth'])))
    verify_inputs(inputs)
    complete=len(windows)==len(planned)
    report=dict(stage='full_depth',status='depth_passed' if complete else 'depth_partial',
        camera_component=manifest['camera_component'],world_scope=manifest['world_scope'],units=UNITS,
        input_frame_count=len(inputs['frames']),unique_frame_count=len(artifacts),
        planned_window_count=len(planned),completed_window_count=len(windows),raw_frame_count=sum(map(len,windows)),
        artifacts=artifacts,windows=reports,overlap_depth_diagnostics=overlaps,checkpoint=checkpoint,
        gpu_uuid=resources['selected_gpu']['uuid'],**check_budget(torch),elapsed_seconds=time.monotonic()-start,
        geometry_accuracy_validated=False,full_pipeline_ready=False,
        limitation='Relative depth; input conditioning E/K are not an accuracy test; no submap pose merging')
    write_json(run/'report.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--camera-component',required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--max-windows',type=int,default=0)
    p.add_argument('--config',default='configs/paths.server.yaml')
    args=p.parse_args()
    if args.max_windows<0:p.error('--max-windows must be nonnegative')
    try:run=create_run_dir(ROOT,args.run_id)
    except (OSError,ValueError) as exc:print(str(exc),file=sys.stderr);return 2
    try:
        report=execute(run,args);write_json(run/'status.json',dict(stage='full_depth',status=report['status'],
            depth_ready=report['status']=='depth_passed',unique_frame_count=report['unique_frame_count']))
        return 0
    except (Exception,KeyboardInterrupt) as exc:
        write_json(run/'status.json',dict(stage='full_depth',status='failed',reason=str(exc),
            error_type=type(exc).__name__,depth_ready=False))
        print(str(exc),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
