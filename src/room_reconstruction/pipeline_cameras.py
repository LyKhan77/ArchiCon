"""Preserve valid views when isolated extreme poses must be quarantined."""
import json
from pathlib import Path
import shutil
import tempfile

from .full_take import trajectory_outliers,prepare_component
from .pipeline_state import fingerprint,workspace_path
from .provenance import sha256_file,write_json


def quarantine_extreme_poses(root,component,job_dir):
    root=Path(root).resolve();component=Path(component).resolve();job_dir=Path(job_dir).resolve()
    if not component.is_relative_to(root/'outputs') or not job_dir.is_relative_to(root/'outputs'):raise ValueError('Camera derivation must remain in outputs')
    names=('status.json','manifest.json','report.json','undistorted/cameras.json')
    hashes={name:sha256_file(component/name) for name in names}
    data={name:json.loads((component/name).read_text()) for name in names}
    camera=data['undistorted/cameras.json'];outliers=trajectory_outliers(camera['frames'])
    if not outliers:return component,[]
    source_status=data['status.json'];source_report=data['report.json']
    if source_status.get('status')!='camera_passed' or not source_report.get('quality',{}).get('passed'):raise ValueError('Cannot derive a rejected camera component')
    excluded={row['frame_id'] for row in outliers};frames=[row for row in camera['frames'] if row['frame_id'] not in excluded]
    if len(frames)<3:raise ValueError('Insufficient valid views after pose quarantine')
    digest=fingerprint(hashes)[:16];destination=job_dir/'camera-quarantine'/digest
    if destination.exists():
        prepare_component(destination)
        return destination,sorted(excluded)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.candidate-',dir=destination.parent) as temporary:
        work=Path(temporary);image_dir=work/'undistorted/images';image_dir.mkdir(parents=True)
        original_images=component/'undistorted'/camera['image_root']
        for frame in frames:
            source=workspace_path(original_images,frame['image_name'])
            if sha256_file(source)!=frame['image_sha256']:raise ValueError('Source image hash differs during quarantine')
            target=image_dir/frame['image_name'];target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
        manifest=dict(data['manifest.json'],selected_frames=[x for x in data['manifest.json']['selected_frames'] if x['frame_id'] not in excluded],
          source_component=str(component.relative_to(root)),source_metadata_sha256=hashes,quarantined_frame_ids=sorted(excluded),
          derivation='Same world and remaining poses; extreme views excluded; no new BA or pose recovery')
        write_json(work/'manifest.json',manifest)
        write_json(work/'undistorted/cameras.json',dict(camera,frames=frames,image_root='images'))
        write_json(work/'report.json',dict(stage='camera_full_component',undistorted=True,registered_images=len(frames),
          quality=dict(passed=True,proper_poses=True,trajectory_outlier_count=0,
                       scope='Subset of accepted source camera component; original residual gate inherited, no new aggregate residual claim'),
          source_component=str(component.relative_to(root)),binary_map_rebuilt=False,quarantined_frame_ids=sorted(excluded)))
        write_json(work/'status.json',dict(stage='camera_full_component',status='camera_passed',camera_ready=True,
                                          quarantined_frame_ids=sorted(excluded),inferred_world_scale='relative; not meters'))
        prepare_component(work)
        if any(sha256_file(component/name)!=digest for name,digest in hashes.items()):raise ValueError('Source metadata changed during derivation')
        # Move complete, checked output to its immutable lineage-addressed location.
        shutil.move(str(work),str(destination))
    prepare_component(destination)
    return destination,sorted(excluded)
