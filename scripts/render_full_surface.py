#!/usr/bin/env python3
"""Render every production camera as actual triangles and export a full-path review video."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import fuse_tsdf as fusion


def review_glb(source,destination,transform):
    from room_reconstruction.mesh_audit import double_sided_glb
    data=source.read_bytes();chunks=[];offset=12
    if struct.unpack_from('<III',data)!=(0x46546c67,2,len(data)):raise ValueError('Expected GLB2')
    while offset<len(data):
        length,kind=struct.unpack_from('<II',data,offset);offset+=8
        chunks.append((kind,data[offset:offset+length]));offset+=length
    doc=json.loads(chunks[0][1]);nodes=doc.setdefault('nodes',[])
    for scene in doc['scenes']:
        old=list(scene.get('nodes',[]));scene['nodes']=[len(nodes)]
        nodes.append(dict(name='Estimated camera up, relative units',matrix=transform.T.ravel().tolist(),children=old))
    raw=json.dumps(doc,separators=(',',':')).encode();raw+=b' '*((-len(raw))%4)
    body=struct.pack('<II',len(raw),chunks[0][0])+raw+b''.join(struct.pack('<II',len(p),k)+p for k,p in chunks[1:])
    output=double_sided_glb(struct.pack('<III',0x46546c67,2,12+len(body))+body)
    # Material and scene-root changes must leave every binary geometry chunk intact.
    pos=12;after=[]
    while pos<len(output):
        length,kind=struct.unpack_from('<II',output,pos);pos+=8;after.append((kind,output[pos:pos+length]));pos+=length
    if chunks[1:]!=after[1:]:raise ValueError('Review export changed binary geometry')
    with destination.open('xb') as stream:stream.write(output)


def display_transform(poses,bounds):
    import numpy as np
    up=-np.asarray(poses)[:,:3,1].mean(0)
    if np.linalg.norm(up)<1e-8:up=-np.asarray(poses)[0,:3,1].copy()
    up/=np.linalg.norm(up)
    positions=np.asarray(poses)[:,:3,3];centered=positions-positions.mean(0)
    horizontal=centered-np.outer(centered@up,up)
    _,_,vt=np.linalg.svd(horizontal,full_matrices=False);x=vt[0]
    x-=np.dot(x,up)*up
    if np.linalg.norm(x)<1e-8:
        x=np.eye(3)[int(np.argmin(np.abs(up)))];x-=np.dot(x,up)*up
    x/=np.linalg.norm(x)
    if np.dot(x,positions[-1]-positions[0])<0:x=-x
    z=np.cross(x,up);z/=np.linalg.norm(z)
    R=np.stack([x,up,z]);T=np.eye(4);T[:3,:3]=R;T[:3,3]=-R@np.mean(bounds,axis=0)
    if not np.allclose(R.T@R,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(R),1):
        raise ValueError('Improper viewer orientation')
    return T


def execute(run,args):
    start=time.monotonic()
    if Path(sys.prefix).resolve()!=ROOT/'.venv-geometry':raise ValueError('Use isolated geometry runtime')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='4'
    fusion.cpu_guard(run)
    import numpy as np
    import open3d as o3d
    from PIL import Image,ImageDraw
    from room_reconstruction.full_take import prepare_component
    from room_reconstruction.depth_da3 import expected_resize
    from room_reconstruction.mesh_audit import TriangleRenderer,pinhole_rays
    if o3d.__version__!='0.19.0' or importlib.metadata.version('open3d')!='0.19.0' or np.__version__!='1.26.4':
        raise ValueError('Validated NumPy/Open3D geometry backend required')
    surface=fusion.run_path(ROOT,args.surface_run)
    state=json.loads((surface/'status.json').read_text());report=json.loads((surface/'report.json').read_text())
    if state.get('stage')!='full_surface' or state.get('status')!='mesh_passed' or report.get('stage')!='full_surface':
        raise ValueError('Accepted full surface required')
    component=(ROOT/report['camera_component']).resolve()
    if not component.is_relative_to(ROOT/'outputs'):raise ValueError('Camera component escapes outputs')
    inputs=prepare_component(component)
    if report['frame_count']!=len(inputs['frames']):raise ValueError('Surface/camera frame scope mismatch')
    hashes=dict(inputs['upstream_hashes'])
    for name in ('status','report','manifest'):hashes[str(surface/f'{name}.json')]=fusion.sha256_file(surface/f'{name}.json')
    selected=report['meshes']['coverage'];artifacts=selected['artifacts']
    def artifact(suffix):
        a=next(a for a in artifacts if a['file'].endswith(suffix));p=(surface/a['file']).resolve()
        if Path(a['file']).is_absolute() or not p.is_relative_to(surface) or fusion.sha256_file(p)!=a['sha256']:
            raise ValueError('Surface artifact path/hash mismatch')
        hashes[str(p)]=a['sha256'];return p
    path=artifact('.ply');glb=artifact('.glb')
    mesh=o3d.io.read_triangle_mesh(str(path));renderer=TriangleRenderer(np.asarray(mesh.vertices),np.asarray(mesh.triangles),np.asarray(mesh.vertex_colors))
    transform=display_transform(inputs['c2w'],np.asarray(selected['bounds']))
    review_glb(glb,run/'room-full.glb',transform)
    (run/'frames').mkdir();records=[];thumbs=[]
    K,hw=expected_resize(inputs['K'],inputs['image_size']);h,w=hw
    for index,(frame,image_path) in enumerate(zip(inputs['frames'],inputs['image_paths'])):
        rays=pinhole_rays(K[index],inputs['c2w'][index],w,h);two,_=renderer.render(rays)
        rgb=np.rint(two['rgb']*255).astype(np.uint8);rgb[~two['hit']]=[32,36,42]
        with Image.open(image_path) as image:original=image.convert('RGB').resize((w,h),Image.Resampling.LANCZOS)
        panel=Image.new('RGB',(w*2,h+28),(245,245,245));panel.paste(original,(0,28));panel.paste(Image.fromarray(rgb),(w,28))
        draw=ImageDraw.Draw(panel);draw.text((5,7),f'{frame["timestamp_seconds"]:.2f}s source',fill=(0,0,0))
        draw.text((w+5,7),f'Triangle render {two["hit"].mean()*100:.1f}% hit',fill=(0,0,0))
        dest=run/'frames'/f'frame_{index:05d}.png';panel.save(dest)
        records.append(dict(frame_id=frame['frame_id'],timestamp_seconds=frame['timestamp_seconds'],
            source_index=frame['source_index'],file=str(dest.relative_to(run)),sha256=fusion.sha256_file(dest),
            triangle_hit_fraction=float(two['hit'].mean()),camera_processed=True,depth_processed=True,
            production_input=True,unbiased_evaluation=False))
        if index%10==0 or index==len(inputs['frames'])-1:
            thumbs.append(panel.resize((280,266)))
        if index%20==0:print(f'Rendered {index+1}/{len(inputs["frames"])} views at {frame["timestamp_seconds"]:.2f}s',flush=True)
        fusion.check_rss()
    sheet=Image.new('RGB',(280*4,266*((len(thumbs)+3)//4)),(245,245,245))
    for index,panel in enumerate(thumbs):sheet.paste(panel,((index%4)*280,(index//4)*266))
    sheet.save(run/'full_path_review.jpg',quality=93)
    # Fixed overview of all triangles, including the useful exterior backface comparison.
    bounds=np.asarray(selected['bounds'])
    corners=np.asarray(np.meshgrid(*zip(bounds[0],bounds[1]))).reshape(3,-1).T
    corners=corners@transform[:3,:3].T+transform[:3,3]
    center=(corners.min(0)+corners.max(0))/2
    span=float(np.linalg.norm(corners.max(0)-corners.min(0)))
    eye=center+np.array([.75,.65,1.])*span
    forward=center-eye;forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right);down=np.cross(forward,right)
    view=np.eye(4);view[:3,:3]=np.column_stack([right,down,forward]);view[:3,3]=eye
    view=np.linalg.inv(transform)@view
    two,one=renderer.render(pinhole_rays([[800,0,480],[0,800,270],[0,0,1]],view,960,540))
    overview=Image.new('RGB',(1920,568),(245,245,245));draw=ImageDraw.Draw(overview)
    for column,(render,label) in enumerate(((one,'Single sided overview'),(two,'Double sided overview'))):
        image=np.rint(render['rgb']*255).astype(np.uint8);image[~render['hit']]=[32,36,42]
        overview.paste(Image.fromarray(image),(column*960,28));draw.text((column*960+5,7),label,fill=(0,0,0))
    overview.save(run/'overview.png')
    video=run/'full-path-review.mp4'
    command=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-framerate','2','-i',str(run/'frames/frame_%05d.png'),
        '-threads','4','-c:v','libx264','-pix_fmt','yuv420p','-crf','22','-movflags','+faststart',str(video)]
    result=subprocess.run(command,check=False,capture_output=True,text=True,timeout=600)
    if result.returncode:raise ValueError('Video encoding failed: '+result.stderr[-2000:])
    fusion.verify_hashes(hashes)
    exported=[run/'room-full.glb',video,run/'full_path_review.jpg',run/'overview.png']
    manifest=dict(stage='full_render',source=fusion.git_snapshot(ROOT),surface_run=args.surface_run,
        camera_component=report['camera_component'],input_and_source_hashes=hashes,
        world_to_viewer=transform.tolist(),viewer_transform='One global proper rigid transform; estimated camera up; no measured gravity; unchanged relative scale',
        geometry_binary_unchanged=True,materials_double_sided=True,
        artifacts=[dict(file=p.name,sha256=fusion.sha256_file(p),size_bytes=p.stat().st_size) for p in exported])
    fusion.write_json(run/'manifest.json',manifest)
    return dict(stage='full_render',status='render_passed',surface_run=args.surface_run,
        frame_count=len(records),frames=records,mean_triangle_hit_fraction=float(np.mean([r['triangle_hit_fraction'] for r in records])),
        observed_timestamp_range_seconds=[records[0]['timestamp_seconds'],records[-1]['timestamp_seconds']],
        all_registered_component_frames_rendered=True,full_video_coverage_claim=False,
        video_fps=2,video_timing='One frame per audited keyframe; timestamps shown; constant-rate presentation differs slightly from irregular source PTS',
        geometry_accuracy_validated=False,units=fusion.UNITS,host_peak_rss_bytes=fusion.check_rss(),elapsed_seconds=time.monotonic()-start)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--surface-run',required=True);p.add_argument('--run-id',required=True)
    args=p.parse_args()
    try:run=fusion.create_run_dir(ROOT,args.run_id)
    except (ValueError,OSError) as exc:print(str(exc),file=sys.stderr);return 2
    try:
        report=execute(run,args);fusion.write_json(run/'report.json',report)
        fusion.write_json(run/'status.json',dict(stage='full_render',status='render_passed',frame_count=report['frame_count']))
        return 0
    except (Exception,KeyboardInterrupt) as exc:
        fusion.write_json(run/'status.json',dict(stage='full_render',status='failed',reason=str(exc),error_type=type(exc).__name__))
        print(str(exc),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
