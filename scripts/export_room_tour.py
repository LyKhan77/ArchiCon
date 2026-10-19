#!/usr/bin/env python3
"""Export draft 2D sections and lighter tour meshes from accepted reconstruction.

Use server .venv-geometry; CPU only. web/build.mjs packages these into offline HTML.
"""
import argparse
import gc
import html
import json
import os
from pathlib import Path
import re
import resource
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from room_reconstruction.provenance import create_run_dir, sha256_file, write_json


def local_file(name):
    path=(ROOT/name).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():raise ValueError('Input must be a workspace file: '+name)
    return path


def memory_guard():
    # Linux reports KiB, macOS bytes. This exporter is designed for the server runtime.
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1024 if sys.platform.startswith('linux') else 1)
    if rss>6*1024**3:raise RuntimeError('Geometry export exceeded 6 GiB host memory budget')
    return rss


def floor_estimate(vertices, faces, camera_heights):
    """Keep incomplete components exportable, with unsupported floors explicit."""
    import numpy as np
    from room_reconstruction.floorplan import estimate_floor
    try:
        floor=estimate_floor(vertices,faces,camera_heights)
        return floor,'area-weighted horizontal face mode below median camera; estimated up, not measured gravity',True
    except ValueError as exc:
        if 'No supported horizontal floor' not in str(exc):raise
        y=np.asarray(vertices,float)[:,1]
        camera_y=float(np.median(camera_heights));span=float(np.ptp(y))
        floor=min(float(np.quantile(y,.05)),camera_y-max(.05*span,1e-5))
        return floor,'unsupported floor: lower geometry percentile below median camera; draft section reference only',False


def export_limitations(config,components):
    coverage=config.get('coverage',{})
    processed=coverage.get('processed',sum(c['stats']['viewpoints'] for c in components))
    attempted=coverage.get('attempted',processed)
    result=['Denah draft dari irisan geometri; furnitur dan partisi dapat muncul sebagai garis.',
            'Belum ada ukuran acuan: unit relatif, bukan meter. Kalibrasi setiap komponen secara terpisah.',
            f'{processed} dari {attempted} keyframe diproses; {len(coverage.get("gaps",[]))} keyframe belum memiliki geometri. Bagian tak terlihat belum direkonstruksi.',
            'HTML memakai mesh yang diringankan; geometri asli tetap tersedia dalam GLB penuh.',
            'Arah atas dan bidang lantai diperkirakan; belum survei dimensi atau denah arsitektur.']
    if len(components)>1:result.append('Komponen terpisah belum disejajarkan; posisi dan skalanya tidak boleh digabung.')
    if any(c.get('floor_supported') is False for c in components):
        result.append('Sebagian komponen tidak memiliki bukti lantai horizontal; irisan memakai referensi bawah geometri yang belum tervalidasi.')
    result.extend(config.get('limitations',[]))
    return list(dict.fromkeys(result))


def viewer_divisor(config):
    import math
    value=float(config.get('viewer_voxel_divisor',450))
    if not math.isfinite(value) or not 100<=value<=2000:
        raise ValueError('Viewer voxel divisor must be finite and between 100 and 2000')
    return value


def drawing(sections, bounds, path, run, prefix):
    import cv2
    import numpy as np
    geometry_lo,geometry_hi=np.asarray(bounds,float)
    span=geometry_hi-geometry_lo
    # Observing cameras can legitimately sit outside the reconstructed surface.
    # Include their path without changing the labelled geometry dimensions.
    points=np.asarray([x['point'] for x in path],float)
    lo=np.minimum(geometry_lo,points.min(0));hi=np.maximum(geometry_hi,points.max(0))
    # Keep line detection resolution tied to geometry, not a distant observer.
    factor=min(1280/max(span),4096/max(hi-lo))
    pad=48;size=np.ceil((hi-lo)*factor).astype(int)+2*pad+1
    def pixels(points):return np.rint((np.asarray(points)-lo)*factor+pad).astype(int)
    layers=[]
    for segments in sections:
        layer=np.zeros((size[1],size[0]),np.uint8)
        for a,b in pixels(segments):cv2.line(layer,tuple(a),tuple(b),255,1,cv2.LINE_AA)
        layers.append(layer)
    evidence=np.maximum.reduce(layers)
    dilated=[cv2.dilate(x,np.ones((5,5),np.uint8)) for x in layers]
    candidates=cv2.HoughLinesP(evidence,1,np.pi/720,threshold=25,
                              minLineLength=max(18,int(factor*.22)),maxLineGap=6)
    accepted=[]
    if candidates is not None:
        for x1,y1,x2,y2 in candidates[:,0]:
            n=max(abs(x2-x1),abs(y2-y1))+1
            xx=np.rint(np.linspace(x1,x2,n)).astype(int);yy=np.rint(np.linspace(y1,y2,n)).astype(int)
            if sum(float((layer[yy,xx]>0).mean())>=.55 for layer in dilated)<2:continue
            a=(np.array([x1,y1])-pad)/factor+lo;b=(np.array([x2,y2])-pad)/factor+lo
            accepted.append([a.tolist(),b.tolist()])
    image=np.full((size[1],size[0],3),(233,241,244),np.uint8)
    image[evidence>20]=[187,182,173]
    for a,b in pixels(accepted) if accepted else []:
        cv2.line(image,tuple(a),tuple(b),(66,63,57),2,cv2.LINE_AA)
    for i,item in enumerate(path):
        if i and not item['break_before']:
            a,b=pixels([path[i-1]['point'],item['point']])
            cv2.line(image,tuple(a),tuple(b),(43,91,191),2,cv2.LINE_AA)
    # Store a background evidence image without path or candidate duplication for interactive SVG.
    background=np.full_like(image,(233,241,244));background[evidence>20]=[187,182,173]
    image_name=prefix+'-evidence.png'
    if not cv2.imwrite(str(run/image_name),background):raise RuntimeError('Could not write evidence PNG')
    png=prefix+'-floorplan.png'
    if not cv2.imwrite(str(run/png),image):raise RuntimeError('Could not write floorplan PNG')
    # Include actual section evidence, candidates and camera path as independent drawing layers.
    import base64
    encoded=base64.b64encode((run/image_name).read_bytes()).decode()
    lines=[]
    for a,b in pixels(accepted) if accepted else []:
        lines.append(f'<path d="M{a[0]} {a[1]}L{b[0]} {b[1]}"/>')
    trail=[]
    for item in path:
        x,y=pixels(item['point']);trail.append(('M' if item['break_before'] else 'L')+f'{x} {y}')
    title=html.escape(prefix+' · draft irisan geometri · skala relatif')
    text=f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size[0]}" height="{size[1]+75}" viewBox="0 0 {size[0]} {size[1]+75}">
<rect width="100%" height="100%" fill="#e9e6df"/><title>{title}</title>
<image href="data:image/png;base64,{encoded}" width="{size[0]}" height="{size[1]}"/>
<g fill="none" stroke="#423f39" stroke-width="2">{''.join(lines)}</g>
<path d="{' '.join(trail)}" fill="none" stroke="#bf5b2b" stroke-width="2"/>
<g font-family="monospace" font-size="13" fill="#423f39"><text x="24" y="{size[1]+22}">{title}</text>
<text x="24" y="{size[1]+44}">Bentang geometri X ≈ {span[0]:.2f} u · Z ≈ {span[1]:.2f} u. Bukan ukuran ruangan.</text>
<text x="24" y="{size[1]+64}">Abu: bukti irisan. Gelap: garis berulang ≥2 ketinggian. Oranye: lintasan kamera.</text></g></svg>'''
    with (run/(prefix+'-floorplan.svg')).open('x') as stream:stream.write(text)
    # Exact mapping of evidence raster to display coordinates includes pixel padding.
    return dict(bounds=[geometry_lo.tolist(),geometry_hi.tolist()],segments=accepted,path=path,
                image_file=image_name,image_bounds=[(lo-pad/factor).tolist(),(lo+(size-1-pad)/factor).tolist()],
                svg_file=prefix+'-floorplan.svg',section_segment_counts=[len(x) for x in sections],
                candidate_policy='Lines supported at ≥55% of their raster samples in at least two of three height sections; walls/furniture not semantically separated')


def execute(config,run):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
    import numpy as np
    import open3d as o3d
    import trimesh
    from room_reconstruction.floorplan import camera_records, estimate_floor, section_segments, colored_scene_mesh
    from room_reconstruction.mesh_audit import double_sided_glb
    if o3d.__version__!='0.19.0':raise ValueError('Use validated Open3D 0.19.0 geometry environment')
    if Path('/proc/meminfo').exists():
        available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
        if available<8*1024**3:raise RuntimeError('Need at least 8 GiB available host RAM before geometry export')
    components=[];input_hashes={};ids=set()
    for c in config['components']:
        prefix=c['id']
        if not re.fullmatch(r'[a-z][a-z0-9_-]{0,39}',prefix) or prefix in ids:raise ValueError('Unique safe component ID required')
        ids.add(prefix)
        source=local_file(c['mesh']);manifest=local_file(c['render_manifest']);camera=local_file(c['cameras'])
        for p in (source,manifest,camera):input_hashes[str(p.relative_to(ROOT))]=sha256_file(p)
        render=json.loads(manifest.read_text())
        digest=input_hashes[c['mesh']]
        if not any(a['sha256']==digest for a in render['artifacts'] if a['file'].endswith('.glb')):
            raise ValueError('Source mesh SHA does not match render provenance')
        # Verify the supplied camera table is exactly the camera table used by the renderer.
        expected=render.get('input_and_source_hashes',{})
        recorded=[v for k,v in expected.items() if k.endswith('/undistorted/cameras.json')]
        if recorded!=[input_hashes[c['cameras']]]:raise ValueError('Camera JSON does not match renderer provenance')
        cameras,path=camera_records(json.loads(camera.read_text())['frames'],render['world_to_viewer'])
        scene=trimesh.load(source,force='scene',process=False)
        mesh=colored_scene_mesh(scene);del scene
        v=np.asarray(mesh.vertices);f=np.asarray(mesh.faces)
        floor,floor_policy,floor_supported=floor_estimate(v,f,[x['position'][1] for x in cameras])
        camera_height=float(np.median([x['position'][1] for x in cameras]))-floor
        if camera_height<=0:raise ValueError('Floor estimate must lie below median camera')
        heights=[floor+camera_height*x for x in (.8,1.2,1.55)]
        bounds=np.array(mesh.bounds)
        sections=[section_segments(v,f,y) for y in heights]
        plan=drawing(sections,bounds[:,[0,2]],path,run,prefix)
        print(prefix,'floor',floor,'section heights',heights,'wall candidates',len(plan['segments']),flush=True)
        write_json(run/(prefix+'-sections.json'),dict(heights=heights,segments=[x.tolist() for x in sections],
                   coordinate_frame='display X/Z, relative units',floor_estimation=floor_policy,floor_supported=floor_supported))
        original_triangles=len(f)
        original_vertices=len(v)
        # Bounded grid reduction for offline HTML; originals remain separately available.
        voxel=float(np.max(bounds[1]-bounds[0])/viewer_divisor(config))
        triangle=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(f))
        color=np.asarray(mesh.visual.vertex_colors)[:,:3]/255.
        triangle.vertex_colors=o3d.utility.Vector3dVector(color)
        reduced=triangle.simplify_vertex_clustering(voxel,o3d.geometry.SimplificationContraction.Average)
        reduced.remove_degenerate_triangles();reduced.remove_duplicated_triangles();reduced.remove_unreferenced_vertices()
        reduced_mesh=trimesh.Trimesh(vertices=np.asarray(reduced.vertices),faces=np.asarray(reduced.triangles),
                                    vertex_colors=np.rint(np.asarray(reduced.vertex_colors)*255).astype(np.uint8),process=False)
        data=double_sided_glb(reduced_mesh.export(file_type='glb'))
        filename=prefix+'-viewer.glb'
        with (run/filename).open('xb') as stream:stream.write(data)
        stats=dict(original_vertices=original_vertices,original_triangles=original_triangles,
                   viewer_vertices=len(reduced.vertices),viewer_triangles=len(reduced.triangles),
                   viewer_bytes=len(data),clustering_voxel_relative=voxel,
                   meters_validated=False,wall_candidates=len(plan['segments']),viewpoints=len(cameras))
        components.append(dict(id=prefix,name=c['name'],mesh_file=filename,bounds=bounds.tolist(),floor_y=floor,
                               ceiling_y=float(bounds[1,1]),cameras=cameras,floorplan=plan,stats=stats,meters_per_unit=None,
                               floor_estimation=floor_policy,floor_supported=floor_supported))
        print(prefix,stats,flush=True)
        del mesh,v,f,triangle,reduced,reduced_mesh,data,sections,color
        gc.collect();memory_guard()
    for name,digest in input_hashes.items():
        if sha256_file(ROOT/name)!=digest:raise ValueError('Input changed while exporting: '+name)
    result=dict(schema_version=1,title=config['title'],source=config['source'],components=components,
                coverage=config.get('coverage',{}),limitations=export_limitations(config,components))
    write_json(run/'tour-data.json',result)
    artifacts=[dict(file=p.name,size_bytes=p.stat().st_size,sha256=sha256_file(p)) for p in sorted(run.iterdir()) if p.is_file()]
    write_json(run/'manifest.json',dict(stage='room_tour_export',input_hashes=input_hashes,artifacts=artifacts))
    write_json(run/'report.json',dict(stage='room_tour_export',status='exported',components=[dict(id=x['id'],**x['stats']) for x in components],
                                    host_peak_rss_bytes=memory_guard(),meters_validated=False,full_video_end_to_end=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/room_tour.json');parser.add_argument('--run-id',required=True)
    args=parser.parse_args();run=None
    try:
        config=json.loads(local_file(args.config).read_text());run=create_run_dir(ROOT,args.run_id)
        execute(config,run)
        write_json(run/'status.json',dict(stage='room_tour_export',status='exported'))
    except Exception as exc:
        if run is not None:write_json(run/'status.json',dict(stage='room_tour_export',status='failed',reason=str(exc)))
        print(str(exc),file=sys.stderr);return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
