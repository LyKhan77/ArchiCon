"""Bounded production contracts; original audit splits remain explicit metadata."""
import json
from pathlib import Path
import re
import numpy as np
from .camera_geometry import validate_pose
from .depth_da3 import expected_resize
from .provenance import sha256_file

UNITS='relative COLMAP world units; not meters'


def trajectory_outliers(frames,radius_ratio=50.):
    """Flag extreme isolated centers; scale-free plausibility check, not accuracy proof."""
    if len(frames)<7:return []
    positions=np.asarray([r['c2w'] for r in frames],float)[:,:3,3]
    if not np.isfinite(positions).all():raise ValueError('Nonfinite camera centers')
    distance=np.linalg.norm(positions-np.median(positions,axis=0),axis=1)
    radius=float(np.quantile(distance,.75))
    if radius<1e-8:return []  # Insufficient spatial baseline; cannot classify reliably.
    return [dict(frame_id=row['frame_id'],timestamp_seconds=row['timestamp_seconds'],
        center_distance=float(distance[i]),reference_radius=radius,center_radius_ratio=float(distance[i]/radius))
        for i,row in enumerate(frames) if distance[i]>radius_ratio*radius]


def plan_windows(count, size=16, stride=12):
    if (isinstance(count,bool) or not isinstance(count,int) or count<3
            or not 3<=size<=16 or not 1<=stride<=size-4):
        raise ValueError('At least three views and bounded overlapping windows required')
    if count<=size:return [list(range(count))]
    starts=list(range(0,count-size+1,stride))
    if starts[-1]!=count-size:starts.append(count-size)
    return [list(range(start,start+size)) for start in starts]


def prepare_component(directory):
    directory=Path(directory).resolve()
    paths={name:directory/f'{name}.json' for name in ('status','manifest','report')}
    paths['cameras']=directory/'undistorted/cameras.json'
    if any(not path.resolve().is_relative_to(directory) for path in paths.values()):
        raise ValueError('Component metadata escapes directory')
    state,manifest,report,meta=[json.loads(paths[k].read_text()) for k in ('status','manifest','report','cameras')]
    if (any(d.get('stage')!='camera_full_component' for d in (state,manifest,report))
            or state.get('status')!='camera_passed' or state.get('camera_ready') is not True
            or report.get('quality',{}).get('passed') is not True or report.get('undistorted') is not True):
        raise ValueError('Accepted undistorted full camera component required')
    if meta.get('schema_version')!=1 or meta.get('pose_convention')!='OpenCV c2w; +x right, +y down, +z forward':
        raise ValueError('Expected global OpenCV camera poses')
    image_root=meta.get('image_root')
    if not isinstance(image_root,str) or not image_root or Path(image_root).is_absolute():
        raise ValueError('Relative image_root required')
    images=(paths['cameras'].parent/image_root).resolve()
    if not images.is_relative_to(paths['cameras'].parent):raise ValueError('Image root escapes component')
    frames=sorted(meta['frames'],key=lambda r:(r['timestamp_seconds'],r['source_index'],r['frame_id']))
    selected=manifest.get('selected_frames',[]);native={r['frame_id']:r for r in selected}
    ids=[r['frame_id'] for r in frames]
    if (len(frames)<3 or len(set(ids))!=len(ids) or len(native)!=len(selected)
            or set(ids)!=set(native) or len({r['source_index'] for r in frames})!=len(ids)):
        raise ValueError('Registered production frame identities must match component manifest')
    outliers=trajectory_outliers(frames)
    if outliers:raise ValueError('Extreme camera trajectory outliers require quarantine: '+','.join(r['frame_id'] for r in outliers))
    paths_list=[];poses=[];intrinsics=[];size=None
    for frame in frames:
        original=native[frame['frame_id']]
        if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',frame['frame_id'])
                or frame.get('audit_split')!=original['split']
                or frame.get('reconstruction_role')!='production' or frame.get('split')!='train'
                or frame['timestamp_seconds']!=original['timestamp_seconds']
                or frame['source_index']!=original['source_index']
                or not np.isfinite(frame['timestamp_seconds']) or frame['timestamp_seconds']<0):
            raise ValueError('Production scope or original audit identity changed')
        name=frame['image_name'];path=(images/name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(images) or sha256_file(path)!=frame['image_sha256']:
            raise ValueError('Undistorted image path/hash mismatch')
        dims=frame['image_size'];K=np.asarray(frame['K'],float)
        if (frame['camera_model'] not in ('PINHOLE','SIMPLE_PINHOLE') or len(dims)!=2 or min(dims)<=0
                or K.shape!=(3,3) or not np.isfinite(K).all() or min(K[0,0],K[1,1])<=0
                or not np.allclose(K[2],[0,0,1]) or abs(K[0,1])+abs(K[1,0])>1e-8):
            raise ValueError('Invalid undistorted pinhole grid')
        if size is not None and size!=dims:raise ValueError('Component must use a shared undistorted grid')
        size=dims
        poses.append(validate_pose(frame['c2w']));intrinsics.append(K);paths_list.append(str(path))
    if len(set(paths_list))!=len(ids):raise ValueError('Duplicate image paths')
    poses=np.asarray(poses,np.float32)
    return dict(frames=frames,image_paths=paths_list,c2w=poses,w2c=np.linalg.inv(poses).astype(np.float32),
        K=np.asarray(intrinsics,np.float32),image_size=size,component_path=str(directory),
        upstream_hashes={str(path):sha256_file(path) for path in paths.values()})


def validate_prediction(prediction,K,w2c,image_size):
    n=len(K);expected,hw=expected_resize(K,image_size)
    d,c=np.asarray(prediction.depth),np.asarray(prediction.conf)
    if (not 3<=n<=16 or len(w2c)!=n or d.shape!=(n,*hw) or c.shape!=d.shape
            or not np.isfinite(c).all() or any(not (np.isfinite(v)&(v>0)).any() for v in d)):
        raise ValueError('Invalid bounded depth/confidence output')
    if (np.asarray(prediction.intrinsics).shape!=(n,3,3)
            or not np.allclose(prediction.intrinsics,expected,atol=2e-4,rtol=0)
            or np.asarray(prediction.extrinsics).shape!=(n,3,4)
            or not np.allclose(prediction.extrinsics,np.asarray(w2c)[:,:3],atol=2e-5,rtol=0)):
        raise ValueError('Processed camera differs from original global conditioning')
    rgb=np.asarray(prediction.processed_images)
    if rgb.shape!=(n,*hw,3) or rgb.dtype!=np.uint8 or bool(prediction.is_metric):
        raise ValueError('RGB grid or relative scale contract mismatch')
    return dict(view_count=n,shape=list(d.shape),units=UNITS,
        valid_depth_fraction=[float((np.isfinite(v)&(v>0)).mean()) for v in d],
        depth_definition='OpenCV optical-axis Z; fixed component global world',
        conditioned_cameras_are_not_accuracy_evidence=True)
