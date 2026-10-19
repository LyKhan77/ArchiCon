#!/usr/bin/env python3
"""CPU person boxes plus reviewed glass polygons; no claim of complete segmentation."""
import argparse
import json
import os
import re
from pathlib import Path
import resource
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from room_reconstruction.pipeline_layout import resolve_run_dir
from room_reconstruction.provenance import create_run_dir, load_yaml, sha256_file, git_snapshot, write_json
from room_reconstruction.resources import ResourcePolicy, evaluate_resources, parse_mem_available


def execute(run, args):
    started = time.monotonic()
    config = load_yaml(ROOT/'configs/paths.server.yaml')
    if Path(config['paths']['project_root']).resolve() != ROOT:
        raise ValueError('Masks require server root')
    resources = evaluate_resources(gpus=[], policy=ResourcePolicy(), selector=None,
        mem_available_bytes=parse_mem_available(Path('/proc/meminfo').read_text()),
        disk_free_bytes=shutil.disk_usage(run).free, busy_uuids=None, gpu_required=False)
    write_json(run/'resources.json', resources)
    if not resources['resource_ready']: raise ValueError('; '.join(resources['errors']))
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', args.camera_run):
        raise ValueError('Unsafe camera run')
    camera = resolve_run_dir(ROOT,args.camera_run)
    if not camera.is_relative_to(ROOT/'outputs'): raise ValueError('Camera run escapes outputs')
    state = json.loads((camera/'status.json').read_text())
    if state.get('status') != 'camera_passed' or state.get('camera_ready') is not True:
        raise ValueError('Accepted training camera run required')
    path = camera/'undistorted/cameras.json'; meta = json.loads(path.read_text())
    native = json.loads((camera/'manifest.json').read_text())
    train = {r['frame_id'] for r in native['selected_frames'] if r['split']=='train'}
    annotation_path = ROOT/args.annotations; annotations = json.loads(annotation_path.read_text())
    weights = ROOT/'models/masking/ssdlite320_mobilenet_v3_large_coco-a79551df.pth'
    digest = sha256_file(weights)
    if not digest.startswith('a79551df'): raise ValueError('Detector weights hash differs from official release')
    image_root = (path.parent/meta['image_root']).resolve()
    if not image_root.is_relative_to(path.parent.resolve()): raise ValueError('Unsafe image root')
    if (len(meta['frames']) != 52 or len(train) != 52
            or {r['frame_id'] for r in meta['frames']} != train or set(annotations['frames']) != train):
        raise ValueError('Expected exact 52-frame annotation coverage for training pilot')
    manifest = {'stage':'semantic_masks','source':git_snapshot(ROOT),'camera_run':args.camera_run,
        'camera_metadata_sha256':sha256_file(path),'annotations_sha256':sha256_file(annotation_path),
        'detector_weights_sha256':digest,'detector':'TorchVision SSDLite320 MobileNetV3 Large COCO_V1',
        'person_score_threshold':.3,'person_box_padding_fraction':.1,'mask_complete':False,
        'uncertain_regions_applied':False,'arguments':vars(args)}
    write_json(run/'manifest.json',manifest)
    import torch
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large
    from torchvision.transforms.functional import to_tensor
    from PIL import Image
    import numpy as np
    from room_reconstruction.semantic_masks import person_boxes, annotation_mask
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    detector=ssdlite320_mobilenet_v3_large(weights=None,weights_backbone=None,num_classes=91).eval()
    detector.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    (run/'masks').mkdir(); records=[]
    with torch.inference_mode():
        for frame in meta['frames']:
            fid=frame['frame_id']; image=(image_root/frame['image_name']).resolve()
            if frame['split']!='train' or fid not in train or not image.is_relative_to(image_root) or sha256_file(image)!=frame['image_sha256']:
                raise ValueError('Invalid training image association')
            entry=annotations['frames'][fid]
            if entry['image_sha256']!=frame['image_sha256']: raise ValueError('Manual annotation image hash changed')
            with Image.open(image) as img:
                img=img.convert('RGB'); size=img.size
                detection=detector([to_tensor(img)])[0]
            boxes=person_boxes(detection['boxes'].numpy(),detection['labels'].numpy(),detection['scores'].numpy(),size)
            boxes+=entry.get('manual_people_boxes',[])
            annotations_for_frame={'people_boxes':boxes,'glass_polygons':entry.get('glass_polygons',[])}
            masks=annotation_mask((504,280),annotations_for_frame)
            artifact=run/'masks'/f'{fid}.npz'
            with artifact.open('xb') as stream:
                np.savez_compressed(stream,**masks)
            records.append({'frame_id':fid,'timestamp_seconds':frame['timestamp_seconds'],'image_sha256':frame['image_sha256'],
                'file':str(artifact.relative_to(run)),'sha256':sha256_file(artifact),**annotations_for_frame,
                'excluded_fraction':float(masks['excluded'].mean()),'person_box_count':len(boxes)})
    if sha256_file(path)!=manifest['camera_metadata_sha256']: raise ValueError('Camera metadata changed')
    report={'stage':'semantic_masks','frames':records,'frame_count':len(records),'mask_shape':[504,280],
        'mask_complete':False,'elapsed_seconds':time.monotonic()-started,'host_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'limitations':['Person boxes are conservative COCO detections plus reviewed annotations; small/occluded people may be missed.',
            'Glass polygons cover visually clear exterior glazing only; uncertain and occluded glass is not fully masked.']}
    write_json(run/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True)
    p.add_argument('--camera-run',default='l2bp3_camera_pilot_002');p.add_argument('--annotations',default='configs/masks_l2bp3.json');a=p.parse_args()
    run=create_run_dir(ROOT,a.run_id)
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[name]='4'
    os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
    try:
        r=execute(run,a);write_json(run/'status.json',{'stage':'semantic_masks','status':'masks_complete','mask_complete':False,'frame_count':r['frame_count']})
        print(f'Masks prepared for {r["frame_count"]} frames: {run}');return 0
    except Exception as e:
        write_json(run/'status.json',{'stage':'semantic_masks','status':'failed','reason':str(e)});print(str(e),file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
