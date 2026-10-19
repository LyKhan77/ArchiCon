#!/usr/bin/env python3
"""Stream CPU coverage/strict surfaces from every registered full-component view."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from room_reconstruction.provenance import create_run_dir,git_snapshot,sha256_file,write_json
from fuse_tsdf import cpu_guard,check_rss,export_mesh,verify_hashes

STAGE='full_surface'
WEIGHTS_SHA256='a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2'


def filter_frames(inputs,run,detect_people,confidence_quantile=.02):
    """Retain only all-frame boolean bases; stream source and one neighbor at a time."""
    import numpy as np
    from room_reconstruction.full_surface import load_depth,nearest_neighbors,coverage_masks,UNITS
    from room_reconstruction.filtering import spatial_masks,view_evidence
    from room_reconstruction.semantic_masks import annotation_mask
    started=time.monotonic();directory=run/'filtered';directory.mkdir()
    semantic=run/'semantic';semantic.mkdir();bases=[];records=[]
    for index,frame in enumerate(inputs['frames']):
        check_rss();fid=frame['frame_id'];depth=load_depth(inputs,index)
        entry=inputs['annotations'][fid];detected=detect_people(index)
        annotations=dict(people_boxes=detected+entry.get('manual_people_boxes',[]),
            glass_polygons=entry.get('glass_polygons',[]))
        masks=annotation_mask(inputs['shape'],annotations)
        spatial=spatial_masks(depth['depth'],depth['conf'],masks['excluded'],confidence_quantile,.08)
        base=spatial['base'];bases.append(base)
        mask_path=semantic/f'{fid}.npz'
        with mask_path.open('xb') as stream:np.savez_compressed(stream,**masks)
        counts={name:int(spatial[key].sum()) for name,key in
            [('valid','valid'),('after_semantic','semantic'),('after_confidence','confidence'),('base','base')]}
        counts.update(total_pixels=base.size,people=int(masks['people'].sum()),glass=int(masks['glass'].sum()),
            semantic_excluded=int(masks['excluded'].sum()),edge_excluded=int((spatial['confidence']&spatial['edge_excluded']).sum()))
        records.append(dict(frame_id=fid,timestamp_seconds=frame['timestamp_seconds'],image_sha256=frame['image_sha256'],
            semantic_file=str(mask_path.relative_to(run)),semantic_sha256=sha256_file(mask_path),
            detector_people_boxes=detected,**annotations,confidence_threshold=spatial['confidence_threshold'],counts=counts))
        del depth,masks,spatial
        print(f'Semantic/base {index+1}/{len(inputs["frames"])}: {fid}',flush=True)
    for index,row in enumerate(records):
        check_rss();source=load_depth(inputs,index);source['valid']=bases[index]
        support=np.zeros(inputs['shape'],np.uint16);contradictions=np.zeros_like(support);occluded=np.zeros_like(support)
        unknown=np.zeros_like(support);neighbors=nearest_neighbors(inputs['frames'],index,4)
        for other in neighbors:
            target=load_depth(inputs,other);target['valid']=bases[other]
            evidence=view_evidence(source,target,.12)
            support+=evidence['support'];contradictions+=evidence['contradiction']
            occluded+=evidence['occluded'];unknown+=evidence['unknown']
            del target,evidence
        masks=coverage_masks(bases[index],support,contradictions)
        path=directory/f'{row["frame_id"]}.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream,base_mask=bases[index],**masks,support=support,
                contradictions=contradictions,occluded=occluded,unknown_neighbor_count=unknown)
        row.update(file=str(path.relative_to(run)),sha256=sha256_file(path),size_bytes=path.stat().st_size,
            neighbor_frame_ids=[inputs['frames'][j]['frame_id'] for j in neighbors])
        row['counts'].update(strict=int(masks['strict_mask'].sum()),coverage=int(masks['coverage_mask'].sum()),
            uncertain=int(masks['uncertain_mask'].sum()),supported=int((bases[index]&(support>0)).sum()),
            contradicted=int((bases[index]&(contradictions>0)).sum()),unsupported=int((bases[index]&(support==0)).sum()),
            rejected_by_contradictions=int((bases[index]&(contradictions>support)).sum()),
            occluded_in_any_neighbor=int((bases[index]&(occluded>0)).sum()),
            unknown_in_any_neighbor=int((bases[index]&(unknown>0)).sum()))
        row['fractions']={key:value/row['counts']['total_pixels'] for key,value in row['counts'].items()}
        del source,support,contradictions,occluded,unknown,masks
        print(f'Multiview {index+1}/{len(records)}: coverage={row["counts"]["coverage"]}, strict={row["counts"]["strict"]}',flush=True)
    report=dict(stage='full_surface_filter',status='filter_passed',frame_count=len(records),frames=records,
        camera_component=inputs['camera_component'],depth_run=inputs['depth_run'],units=UNITS,
        counts={key:sum(row['counts'][key] for row in records) for key in records[0]['counts']},
        settings=dict(confidence_quantile=confidence_quantile,edge_relative=.08,nearest_time_neighbors=4,
            relative_tolerance=.12,strict='base & (support >= 1) & (support >= contradictions)',
            coverage='base & (support >= contradictions)',uncertain='base & (support == 0) & (contradictions == 0)'),
        mask_complete=False,elapsed_seconds=time.monotonic()-started,
        limitations=['Person boxes and reviewed glazing polygons may miss small, hidden, or uncertain semantic regions.',
            'Unsupported coverage is explicitly uncertain; retained observed depths are never hole filling.'])
    write_json(directory/'report.json',report)
    write_json(directory/'manifest.json',dict(stage='full_surface_filter',camera_component=inputs['camera_component'],
        depth_run=inputs['depth_run'],input_and_source_hashes=inputs['input_and_source_hashes']))
    write_json(directory/'status.json',dict(stage='full_surface_filter',status='filter_passed',frame_count=len(records)))
    return report


def iter_filtered(inputs,run,filter_report,mask_key):
    import numpy as np
    from room_reconstruction.full_surface import load_depth,coverage_masks
    for index,row in enumerate(filter_report['frames']):
        check_rss();path=run/row['file']
        if sha256_file(path)!=row['sha256']:raise ValueError('Filter checkpoint hash changed')
        frame=load_depth(inputs,index)
        with np.load(path,allow_pickle=False) as archive:
            base=archive['base_mask'];mask=archive[mask_key]
            evidence=coverage_masks(base,archive['support'],archive['contradictions'])
        if (base.dtype!=bool or mask.dtype!=bool or base.shape!=inputs['shape'] or mask.shape!=inputs['shape']
                or not np.array_equal(mask,evidence[mask_key])
                or np.any(base&(~np.isfinite(frame['depth'])|(frame['depth']<=0)))):
            raise ValueError('Invalid filtered mask contract')
        yield frame,mask


def detector_callback(inputs,weights):
    import torch
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large
    from torchvision.transforms.functional import to_tensor
    from PIL import Image
    from room_reconstruction.semantic_masks import person_boxes
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    detector=ssdlite320_mobilenet_v3_large(weights=None,weights_backbone=None,num_classes=91).eval()
    detector.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    def detect(index):
        with Image.open(inputs['image_paths'][index]) as image:
            image=image.convert('RGB');size=image.size
            if list(size)!=inputs['image_size']:raise ValueError('Decoded RGB size differs from component camera')
            with torch.inference_mode():detection=detector([to_tensor(image)])[0]
        return person_boxes(detection['boxes'].numpy(),detection['labels'].numpy(),detection['scores'].numpy(),size,.3,.1)
    return detect


def execute(root,run,args):
    started=time.monotonic();root=Path(root).resolve()
    if Path(sys.prefix).resolve()!=root/'.venv-geometry':raise ValueError('Use isolated .venv-geometry/bin/python')
    if any(name in sys.modules for name in ('torch','open3d')):raise ValueError('Heavy library imported before CPU resource gate')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='4'
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode=True
    resources=cpu_guard(run);check_rss()
    import numpy as np
    from room_reconstruction.full_surface import prepare_inputs,scene_parameters,UNITS
    from room_reconstruction.meshing import integrate_mesh
    if np.__version__!='1.26.4':raise ValueError('Validated NumPy 1.26.4 required')
    version=importlib.metadata.version('open3d')
    if version!='0.19.0':raise ValueError(f'Validated Open3D 0.19.0 required; found {version}')
    inputs=prepare_inputs(root,args.depth_run,args.camera_component,root/args.annotations)
    weights=root/'models/masking/ssdlite320_mobilenet_v3_large_coco-a79551df.pth'
    if sha256_file(weights)!=WEIGHTS_SHA256:raise ValueError('Detector weights SHA256 differs from pinned official checkpoint')
    hashes=inputs['input_and_source_hashes'];hashes[str(weights)]=WEIGHTS_SHA256
    for relative in ('scripts/build_full_surface.py','scripts/fuse_tsdf.py','src/room_reconstruction/full_surface.py',
            'src/room_reconstruction/full_take.py','src/room_reconstruction/depth_da3.py',
            'src/room_reconstruction/filtering.py','src/room_reconstruction/semantic_masks.py',
            'src/room_reconstruction/meshing.py','src/room_reconstruction/camera_geometry.py'):
        path=root/relative;hashes[str(path)]=sha256_file(path)
    manifest=dict(stage=STAGE,source=git_snapshot(root),arguments=vars(args),camera_component=inputs['camera_component'],
        depth_run=args.depth_run,selected_frames=inputs['frames'],input_and_source_hashes=hashes,units=UNITS,
        world_scope=inputs['camera_component'],production_uses_original_validation=True,
        detector='TorchVision SSDLite320 MobileNetV3 Large COCO_V1; CPU; score >= 0.3; box padding 0.1',
        detector_weights_sha256=WEIGHTS_SHA256,cpu_threads=4,geometry_accuracy_validated=False)
    write_json(run/'manifest.json',manifest)
    detect=detector_callback(inputs,weights)
    filter_report=filter_frames(inputs,run,detect,args.confidence_quantile);del detect
    import gc
    gc.collect();check_rss()
    # Parameters use the same spatial baseline for both surfaces, never per-mode tuning.
    def values():
        for index,row in enumerate(filter_report['frames']):
            from room_reconstruction.full_surface import load_depth
            frame=load_depth(inputs,index)
            with np.load(run/row['file'],allow_pickle=False) as archive:base=archive['base_mask']
            yield frame['depth'][base]
    parameters=scene_parameters(values(),args.voxel_divisor,args.sdf_voxels)
    # Report the exact additional clipping performed by TSDF, per view and mode.
    from room_reconstruction.full_surface import load_depth
    for index,row in enumerate(filter_report['frames']):
        frame=load_depth(inputs,index);inside=frame['depth']<parameters['depth_trunc']
        with np.load(run/row['file'],allow_pickle=False) as archive:
            row['counts'].update(coverage_integrated=int((archive['coverage_mask']&inside).sum()),
                strict_integrated=int((archive['strict_mask']&inside).sum()),
                depth_trunc_excluded_base=int((archive['base_mask']&~inside).sum()))
        row['fractions']={key:value/row['counts']['total_pixels'] for key,value in row['counts'].items()}
        del frame,inside
    filter_report['counts']={key:sum(row['counts'][key] for row in filter_report['frames'])
        for key in filter_report['frames'][0]['counts']}
    filter_report['integration_parameters']=parameters
    write_json(run/'filtered/integration_report.json',filter_report)
    write_json(run/'parameters.json',dict(stage=STAGE,camera_component=inputs['camera_component'],
        depth_run=args.depth_run,parameters=parameters))
    import open3d as o3d
    if o3d.__version__!='0.19.0':raise ValueError('Imported backend differs from validated Open3D 0.19.0')
    meshes={};settings={key:parameters[key] for key in ('voxel_length','sdf_trunc','depth_trunc')}
    for name,mask_key in [('coverage','coverage_mask'),('strict','strict_mask')]:
        if name=='strict' and args.skip_strict:
            meshes[name]=dict(status='skipped_by_request');continue
        print(f'Integrating {name}: {len(inputs["frames"])} production frames in fixed component world',flush=True)
        mesh=integrate_mesh(iter_filtered(inputs,run,filter_report,mask_key),**settings);check_rss()
        if name=='strict' and (not len(mesh.vertices) or not len(mesh.triangles)):
            meshes[name]=dict(status='empty',vertex_count=len(mesh.vertices),triangle_count=len(mesh.triangles))
        else:
            meshes[name]=dict(status='mesh_passed',**export_mesh(mesh,run,name))
            if not any(row['file'].endswith('.glb') for row in meshes[name]['artifacts']):
                raise ValueError(f'{name} GLB export required')
        del mesh;gc.collect()
    verify_hashes(hashes)
    for row in filter_report['frames']:
        if sha256_file(run/row['file'])!=row['sha256'] or sha256_file(run/row['semantic_file'])!=row['semantic_sha256']:
            raise ValueError('Filter or semantic artifact changed during fusion')
    return dict(stage=STAGE,status='mesh_passed',frame_count=len(inputs['frames']),camera_component=inputs['camera_component'],
        depth_run=args.depth_run,meshes=meshes,primary_mesh='coverage',parameters=parameters,units=UNITS,
        frames=filter_report['frames'],counts=filter_report['counts'],filter_checkpoint='filtered',
        semantic_mask_complete=False,geometry_accuracy_validated=False,full_pipeline_ready=False,
        component_policy='All components retained; no smoothing, hole filling, or component pruning',
        coverage_policy='Observed unsupported noncontradicted depths retained and labelled uncertain',
        runtime_versions=dict(numpy=np.__version__,open3d=version,torch=importlib.metadata.version('torch'),
            torchvision=importlib.metadata.version('torchvision')),resources=resources,
        host_peak_rss_bytes=check_rss(),elapsed_seconds=time.monotonic()-started)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--depth-run',required=True);parser.add_argument('--camera-component',required=True)
    parser.add_argument('--annotations',default='configs/masks_l2bp3_full.json');parser.add_argument('--run-id',required=True)
    parser.add_argument('--confidence-quantile',type=float,default=.02)
    parser.add_argument('--voxel-divisor',type=float,default=100);parser.add_argument('--sdf-voxels',type=float,default=8)
    parser.add_argument('--skip-strict',action='store_true',help='Skip optional strict comparison surface')
    args=parser.parse_args(argv)
    if not 0<=args.confidence_quantile<1:parser.error('--confidence-quantile must be in [0,1)')
    if not 0<args.voxel_divisor<float('inf') or not 1<args.sdf_voxels<float('inf'):
        parser.error('Finite positive voxel divisor and SDF truncation above one voxel required')
    try:run=create_run_dir(ROOT.resolve(),args.run_id)
    except (OSError,ValueError) as exc:print(str(exc),file=sys.stderr);return 2
    try:
        report=execute(ROOT.resolve(),run,args);write_json(run/'report.json',report)
        write_json(run/'status.json',dict(stage=STAGE,status='mesh_passed',mesh_ready=True,
            frame_count=report['frame_count'],geometry_accuracy_validated=False,full_pipeline_ready=False))
        return 0
    except (Exception,KeyboardInterrupt) as exc:
        failure=dict(stage=STAGE,status='failed',reason=str(exc),error_type=type(exc).__name__,mesh_ready=False,arguments=vars(args))
        for name in ('manifest','report','status'):
            if not (run/f'{name}.json').exists():write_json(run/f'{name}.json',failure)
        print(f'Full surface failed: {type(exc).__name__}: {exc}',file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
