"""Production full-component surface contracts and explicit uncertain coverage."""
import json
from pathlib import Path
import re

import numpy as np

from .depth_da3 import expected_resize
from .full_take import prepare_component, UNITS
from .provenance import sha256_file
from .pipeline_layout import resolve_run_dir


def coverage_masks(base, support, contradictions):
    base=np.asarray(base,bool);support=np.asarray(support);contradictions=np.asarray(contradictions)
    if base.ndim!=2 or support.shape!=base.shape or contradictions.shape!=base.shape:
        raise ValueError('Evidence must share the base mask grid')
    if (not np.issubdtype(support.dtype,np.integer) or not np.issubdtype(contradictions.dtype,np.integer)
            or (support<0).any() or (contradictions<0).any()):
        raise ValueError('Evidence counts must be nonnegative integers')
    coverage=base & (support>=contradictions)
    return dict(coverage_mask=coverage,strict_mask=coverage & (support>=1),
        uncertain_mask=base & (support==0) & (contradictions==0))


def nearest_neighbors(frames,index,count=4):
    return sorted((j for j in range(len(frames)) if j!=index),
        key=lambda j:(abs(frames[j]['timestamp_seconds']-frames[index]['timestamp_seconds']),j))[:count]


def prepare_inputs(root,depth_run,camera_component,annotations_path):
    root=Path(root).resolve();outputs=(root/'outputs').resolve()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',depth_run):
        raise ValueError('Unsafe depth run ID')
    directory=resolve_run_dir(root,depth_run);component=(root/camera_component).resolve()
    if (Path(camera_component).is_absolute() or not component.is_relative_to(outputs)
            or not directory.is_relative_to(outputs)):
        raise ValueError('Camera component and depth run must remain within outputs')
    camera=prepare_component(component)
    meta=[directory/f'{name}.json' for name in ('status','manifest','report')]
    if any(not p.resolve().is_relative_to(directory) for p in meta):raise ValueError('Depth metadata escapes run')
    state,manifest,report=[json.loads(p.read_text()) for p in meta]
    relative=str(component.relative_to(root));frames=camera['frames'];n=len(frames)
    if (any(v.get('stage')!='full_depth' for v in (state,manifest,report))
            or state.get('status')!='depth_passed' or state.get('depth_ready') is not True
            or report.get('status')!='depth_passed' or report.get('units')!=UNITS
            or report.get('unique_frame_count')!=n
            or any(v.get('camera_component')!=relative for v in (manifest,report))
            or manifest.get('selected_frames')!=frames):
        raise ValueError('Complete full_depth stage with identical global component frames required')
    rows=report.get('artifacts',[]);ids=[r['frame_id'] for r in rows]
    if len(rows)!=n or len(set(ids))!=n or set(ids)!={r['frame_id'] for r in frames}:
        raise ValueError('Depth artifacts must cover each component frame exactly once')
    hashes=dict(camera['upstream_hashes']);hashes.update({str(p):sha256_file(p) for p in meta})
    hashes.update({path:row['image_sha256'] for path,row in zip(camera['image_paths'],frames,strict=True)})
    for name,digest in manifest.get('upstream_hashes',{}).items():
        path=Path(name).resolve()
        if not path.is_relative_to(outputs) or sha256_file(path)!=digest:
            raise ValueError('Depth upstream path/hash mismatch')
        if str(path) in hashes and hashes[str(path)]!=digest:
            raise ValueError('Depth was produced from a different camera metadata version')
        hashes[str(path)]=digest
    by_id={r['frame_id']:r for r in rows};paths=[];ordered=[]
    for frame in frames:
        row=by_id[frame['frame_id']];path=(directory/row['file']).resolve()
        if (Path(row['file']).is_absolute() or not path.is_relative_to(directory)
                or row.get('timestamp_seconds')!=frame['timestamp_seconds']
                or sha256_file(path)!=row['sha256']):
            raise ValueError('Depth artifact path, identity, or hash mismatch')
        paths.append(path);ordered.append(row);hashes[str(path)]=row['sha256']
    if len(set(paths))!=n:raise ValueError('Duplicate depth artifact paths')
    annotations_path=Path(annotations_path).resolve()
    annotations=json.loads(annotations_path.read_text()).get('frames',{})
    if set(annotations)!={r['frame_id'] for r in frames}:
        raise ValueError('Annotations require exact full component frame coverage')
    for frame in frames:
        if annotations[frame['frame_id']].get('image_sha256')!=frame['image_sha256']:
            raise ValueError('Annotation image hash differs from undistorted component image')
    hashes[str(annotations_path)]=sha256_file(annotations_path)
    K,hw=expected_resize(camera['K'],camera['image_size'])
    return dict(camera,paths=paths,artifacts=ordered,annotations=annotations,
        camera_component=relative,depth_run=depth_run,expected_K=K,shape=hw,input_and_source_hashes=hashes)


def load_depth(inputs,index):
    path=inputs['paths'][index]
    if sha256_file(path)!=inputs['artifacts'][index]['sha256']:
        raise ValueError('Depth artifact changed after validation')
    with np.load(path,allow_pickle=False) as archive:
        frame={key:archive[key] for key in ('depth','conf','K','w2c','input_c2w','processed_rgb')}
    depth,conf=frame['depth'],frame['conf'];hw=inputs['shape']
    if (depth.shape!=hw or depth.dtype!=np.float32 or conf.shape!=hw or conf.dtype!=np.float32
            or not np.isfinite(conf).all() or not (np.isfinite(depth)&(depth>0)).any()):
        raise ValueError('Invalid full depth/confidence schema')
    if (frame['K'].shape!=(3,3) or frame['w2c'].shape!=(3,4) or frame['input_c2w'].shape!=(4,4)
            or not np.allclose(frame['K'],inputs['expected_K'][index],rtol=0,atol=2e-4)
            or not np.allclose(frame['w2c'],inputs['w2c'][index,:3],rtol=0,atol=2e-5)
            or not np.allclose(frame['input_c2w'],inputs['c2w'][index],rtol=0,atol=2e-5)):
        raise ValueError('Depth camera differs from original global component camera')
    if frame['processed_rgb'].shape!=(*hw,3) or frame['processed_rgb'].dtype!=np.uint8:
        raise ValueError('Invalid processed RGB grid')
    return frame


def scene_parameters(frame_values,voxel_divisor=100,sdf_voxels=8):
    if (not np.isfinite([voxel_divisor,sdf_voxels]).all() or voxel_divisor<=0 or sdf_voxels<=1):
        raise ValueError('Positive voxel divisor and SDF truncation above one voxel required')
    samples=[]
    for values in frame_values:
        values=np.asarray(values).ravel();values=values[np.isfinite(values)&(values>0)]
        if len(values):samples.append(values[::max(1,(len(values)+4095)//4096)].copy())
    if not samples:raise ValueError('No retained positive depths for scene parameters')
    values=np.concatenate(samples);median=float(np.median(values));voxel=median/voxel_divisor
    return dict(voxel_length=voxel,sdf_trunc=sdf_voxels*voxel,
        depth_trunc=1.1*float(np.quantile(values,.995)),sample_count=len(values),
        sample_policy='Deterministic stride; at most 4096 positive baseline depths per frame',
        median_depth=median,voxel_divisor=voxel_divisor,sdf_voxels=sdf_voxels,units=UNITS)
