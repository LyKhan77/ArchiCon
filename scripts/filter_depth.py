#!/usr/bin/env python3
"""CPU staged masking/occlusion-aware filtering, with explicit coverage losses."""
import argparse
import json
import os
from pathlib import Path
import re
import resource
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from room_reconstruction.pipeline_layout import resolve_run_dir
from room_reconstruction.provenance import create_run_dir,load_yaml,sha256_file,git_snapshot,write_json
from room_reconstruction.resources import ResourcePolicy,evaluate_resources,parse_mem_available


def read_stage(root,name,stage,status):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',name):raise ValueError('Unsafe input run ID')
    run=resolve_run_dir(root,name)
    if not run.is_relative_to((root/'outputs').resolve()):raise ValueError('Input run escapes outputs')
    state=json.loads((run/'status.json').read_text());report=json.loads((run/'report.json').read_text())
    if state.get('stage')!=stage or state.get('status')!=status or report.get('stage')!=stage:
        raise ValueError(f'Input {name} is not completed {stage}/{status}')
    return run,report


def checked_files(run,records):
    result={}
    for row in records:
        fid=row['frame_id'];path=(run/row['file']).resolve()
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',fid) or fid in result:
            raise ValueError('Unsafe or duplicate frame ID')
        if not path.is_relative_to(run) or sha256_file(path)!=row['sha256']:
            raise ValueError('Artifact path/hash mismatch')
        result[fid]=(path,row)
    if len(result)!=52:raise ValueError('Exactly 52 unique training artifacts required')
    return result


def execute(run,args):
    started=time.monotonic();config=load_yaml(ROOT/'configs/paths.server.yaml')
    if Path(config['paths']['project_root']).resolve()!=ROOT:raise ValueError('Server root required')
    resources=evaluate_resources(gpus=[],policy=ResourcePolicy(),selector=None,
        mem_available_bytes=parse_mem_available(Path('/proc/meminfo').read_text()),
        disk_free_bytes=shutil.disk_usage(run).free,busy_uuids=None,gpu_required=False)
    write_json(run/'resources.json',resources)
    if not resources['resource_ready']:raise ValueError('; '.join(resources['errors']))
    depth,dreport=read_stage(ROOT,args.depth_run,'depth_pilot','depth_passed')
    masks,mreport=read_stage(ROOT,args.mask_run,'semantic_masks','masks_complete')
    dm=json.loads((depth/'manifest.json').read_text());mm=json.loads((masks/'manifest.json').read_text())
    if dm['camera_run']!=mm['camera_run']:raise ValueError('Mask/depth camera runs differ')
    rows=sorted(dm['selected_train_frames'],key=lambda r:r['timestamp_seconds'])
    if len(rows)!=52 or any(r['split']!='train' for r in rows) or len({r['frame_id'] for r in rows})!=52:
        raise ValueError('Expected 52 unique TRAIN frames')
    files=checked_files(depth,dreport['artifacts']);maskfiles=checked_files(masks,mreport['frames'])
    if set(files)!=set(maskfiles) or set(files)!={r['frame_id'] for r in rows}:raise ValueError('Frame sets differ')
    upstream={str(p):sha256_file(p) for folder in (depth,masks) for p in [folder/'status.json',folder/'report.json',folder/'manifest.json']}
    write_json(run/'manifest.json',{'stage':'filtered_depth','source':git_snapshot(ROOT),'arguments':vars(args),
        'camera_run':dm['camera_run'],'depth_run':args.depth_run,'mask_run':args.mask_run,
        'selected_train_frames':rows,'upstream_hashes':upstream,
        'settings':{'confidence_quantile':args.confidence_quantile,'edge_relative':.08,'relative_multiview_tolerance':.12,
            'nearest_temporal_neighbors':4,'minimum_support':1,'support_must_equal_or_exceed_contradictions':True},
        'mask_complete':False,'units':'relative COLMAP world units; not meters'})
    import numpy as np
    from room_reconstruction.filtering import spatial_masks,multiview_mask,depth_normals
    from room_reconstruction.depth_da3 import expected_resize
    spatial={};stats=[]
    def load(fid):
        path,_=files[fid]
        with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in ('depth','conf','K','w2c','input_c2w','processed_rgb')}
    # Small boolean cache only; prediction arrays loaded one source/neighbour at a time.
    for row in rows:
        fid=row['frame_id'];f=load(fid)
        if maskfiles[fid][1]['image_sha256']!=row['image_sha256']:raise ValueError('Mask references another input image')
        if f['depth'].shape!=(504,280) or f['conf'].shape!=f['depth'].shape or f['processed_rgb'].shape!=(504,280,3):raise ValueError('Wrong processed grid')
        expected,_=expected_resize(np.array([row['K']]),row['image_size'])
        if not np.allclose(f['K'],expected[0],atol=2e-4,rtol=0) or not np.allclose(f['input_c2w'],row['c2w'],atol=2e-5,rtol=0):raise ValueError('Processed camera differs from original global camera')
        if not np.allclose(f['w2c'],np.linalg.inv(f['input_c2w'])[:3],atol=2e-5,rtol=0):raise ValueError('w2c/input pose mismatch')
        with np.load(maskfiles[fid][0],allow_pickle=False) as a:excluded=a['excluded'].astype(bool)
        stages=spatial_masks(f['depth'],f['conf'],excluded,confidence_quantile=args.confidence_quantile)
        spatial[fid]=stages['base']
        stats.append({'frame_id':fid,'timestamp_seconds':row['timestamp_seconds'],
            'valid_fraction':float(stages['valid'].mean()),'after_semantic_fraction':float(stages['semantic'].mean()),
            'after_confidence_fraction':float(stages['confidence'].mean()),'after_edges_fraction':float(stages['base'].mean()),
            'confidence_threshold':stages['confidence_threshold']})
    artifacts=[]
    for row,stat in zip(rows,stats,strict=True):
        fid=row['frame_id'];f=load(fid);f['valid']=spatial[fid]
        neighbors=sorted((r for r in rows if r['frame_id']!=fid),key=lambda r:abs(r['timestamp_seconds']-row['timestamp_seconds']))[:4]
        loaded=[]
        for neighbor in neighbors:
            nf=load(neighbor['frame_id']);nf['valid']=spatial[neighbor['frame_id']];loaded.append(nf)
        mv=multiview_mask(f,loaded);normal,nvalid=depth_normals(f['depth'],f['K'],mv['mask'])
        artifact=run/f'{fid}.npz'
        with artifact.open('xb') as stream:
            np.savez_compressed(stream,**{k:f[k] for k in ('depth','conf','K','w2c','processed_rgb')},
                base_mask=spatial[fid],mask=mv['mask'],normal=normal,normal_valid=nvalid,
                support=mv['support'],contradictions=mv['contradictions'],occluded=mv['occluded'])
        fraction=float(mv['mask'].mean());base=float(spatial[fid].mean())
        stat.update(after_multiview_fraction=fraction,retained_base_fraction=fraction/base if base else 0,
            unsupported_fraction=float(mv['unsupported'].mean()),occluded_any_fraction=float((mv['occluded']>0).mean()),
            contradiction_any_fraction=float((mv['contradictions']>0).mean()),neighbor_frame_ids=[r['frame_id'] for r in neighbors])
        artifacts.append(dict(stat,file=artifact.name,sha256=sha256_file(artifact)))
    for p,digest in upstream.items():
        if sha256_file(Path(p))!=digest:raise ValueError('Upstream stage metadata changed')
    report={'stage':'filtered_depth','depth_run':args.depth_run,'mask_run':args.mask_run,'camera_run':dm['camera_run'],
        'frame_count':52,'frames':artifacts,'units':'relative COLMAP world units; not meters',
        'normal_convention':'OpenCV camera-facing unit normals, -Z on a fronto-parallel plane',
        'mean_base_coverage':float(np.mean([r['after_edges_fraction'] for r in stats])),
        'mean_filtered_coverage':float(np.mean([r['after_multiview_fraction'] for r in stats])),
        'coverage_flags':[r['frame_id'] for r in stats if r['retained_base_fraction']<.5],
        'mask_complete':False,'geometry_accuracy_validated':False,'elapsed_seconds':time.monotonic()-started,
        'host_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'limitations':['No-support and occlusion are separately labelled, not counted as contradictions.',
            'TSDF base and filtered meshes must be compared to inspect coverage losses; semantic masks remain incomplete.']}
    write_json(run/'report.json',report);return report


def confidence_quantile(value):
    q=float(value)
    if not 0<=q<1:raise argparse.ArgumentTypeError('confidence quantile must satisfy 0 <= q < 1')
    return q


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--depth-run',required=True);p.add_argument('--mask-run',required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--confidence-quantile',type=confidence_quantile,default=.1);a=p.parse_args(argv)
    try:run=create_run_dir(ROOT,a.run_id)
    except (OSError,ValueError) as e:print(str(e),file=sys.stderr);return 2
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='4'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    try:
        r=execute(run,a);write_json(run/'status.json',{'stage':'filtered_depth','status':'filter_passed','frame_count':52,'geometry_accuracy_validated':False})
        print(f'Coverage base={r["mean_base_coverage"]:.3f}, filtered={r["mean_filtered_coverage"]:.3f}');return 0
    except Exception as e:
        write_json(run/'status.json',{'stage':'filtered_depth','status':'failed','reason':str(e)});print(str(e),file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
