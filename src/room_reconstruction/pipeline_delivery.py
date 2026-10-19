"""Verified, versioned products; PART1 and reconstruction artifacts stay immutable."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

from .pipeline_state import atomic_json,workspace_path
from .provenance import sha256_file
from .pipeline_layout import resolve_run_dir,validate_job_id


def verify_bundle(root,bundle,*,input_sha256):
    if bundle.get('schema_version')!=1 or bundle.get('input_sha256')!=input_sha256:raise ValueError('Bundle video identity differs')
    names=[]
    for item in bundle['artifacts']:
        path=workspace_path(root,item['file'])
        if sha256_file(path)!=item['sha256'] or path.stat().st_size!=item['size_bytes']:raise ValueError('Bundle artifact checksum differs: '+item['file'])
        names.append(item['file'])
    if len(set(names))!=len(names):raise ValueError('Duplicate bundle artifacts')
    return bundle


def _copy(source,destination):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    digest=sha256_file(source)
    with Path(source).open('rb') as reader,destination.open('xb') as writer:shutil.copyfileobj(reader,writer,8*1024**2)
    if sha256_file(destination)!=digest:raise RuntimeError('Delivery transfer checksum differs')


def make_delivery(root,job_dir,bundle,*,delivery_root='end_product'):
    validate_job_id(bundle['job_id'])
    root=Path(root).resolve();base=workspace_path(root,delivery_root,must_exist=False)/bundle['job_id']
    if not base.resolve().is_relative_to(root/'end_product') or base.resolve().is_relative_to(root/'end_product/part1'):
        raise ValueError('Delivery must remain in end_product outside immutable PART1')
    base.mkdir(parents=True,exist_ok=True);number=1
    while (base/f'v{number:03d}').exists():number+=1
    delivery=base/f'v{number:03d}';delivery.mkdir()
    export=resolve_run_dir(root,bundle['tour_run']);data=json.loads((export/'tour-data.json').read_text())
    for component in bundle['components']:
        identifier=component['id']
        if '/' in identifier or '\\' in identifier or identifier in ('.','..'):raise ValueError('Unsafe component ID')
        _copy(workspace_path(root,component['mesh']),delivery/'mesh'/f'{identifier}.glb')
        info=next(x for x in data['components'] if x['id']==identifier)
        for key in ('svg_file',):
            name=info['floorplan'].get(key)
            if name and (export/name).is_file():_copy(workspace_path(export,name),delivery/'denah'/f'{identifier}.svg')
        image=export/(identifier+'-floorplan.png')
        if image.is_file():_copy(image,delivery/'denah'/f'{identifier}.png')
    (delivery/'room_tour').mkdir()
    result=subprocess.run(['node',str(root/'web/build.mjs'),'--data',str(export/'tour-data.json'),
                           '--output',str(delivery/'room_tour/index.html')],cwd=root,capture_output=True,text=True,check=False)
    # Parent directory is intentionally created explicitly; builder refuses output overwrite.
    if result.returncode:raise RuntimeError('HTML packaging failed: '+result.stderr[-1800:])
    atomic_json(delivery/'reports/coverage.json',bundle['coverage'])
    atomic_json(delivery/'reports/component-failures.json',bundle.get('failed_components',[]))
    atomic_json(delivery/'calibration.json',dict(schema_version=1,scale_status='relative',components=[]))
    return delivery


def review_packet(root,job_dir,bundle,delivery,round_number=0,candidates=None):
    root=Path(root).resolve();qa=list(bundle.get('qa',[]))
    # Include product views after source/render evidence; all paths must be actual bundle or QA outputs.
    for path in delivery.rglob('*.png'):qa.append(str(path.relative_to(root)))
    qa=list(dict.fromkeys(qa))
    for path in qa:workspace_path(root,path)
    return dict(schema_version=1,job_id=bundle['job_id'],review_round=round_number,
      component_ids=[x['id'] for x in bundle['components']],coverage=bundle['coverage'],scale_status='relative',
      failed_components=bundle.get('failed_components',[]),component_evidence=bundle.get('qa_component_evidence',{}),
      evidence=qa,delivery=str(delivery.relative_to(root)),candidate_ids=['baseline',*(candidates or {}).keys()],candidates=candidates or {},
      limitations=['Relative units; no measured reference.','Separate components are not spatially aligned.',
                  'Temporal/frame coverage is not complete interior surface coverage.','Mild holes tolerated; missing observed areas must be reported.'])


def finalize_delivery(root,delivery,bundle,review,review_dir,*,pending=False):
    root=Path(root).resolve();delivery=Path(delivery)
    if (delivery/'manifest.json').exists():raise FileExistsError('Delivery version already finalized')
    state='pending' if pending else 'unavailable';quality='unassessed'
    if review is not None:
        state={'accept':'passed','refine':'refine_requested','needs_capture':'needs_capture','needs_human_review':'needs_human_review'}[review['decision']]
        quality=review['quality_status']
        atomic_json(delivery/'reports/agent-review.json',review)
        if review_dir and (Path(review_dir)/'agent-review.md').is_file():_copy(Path(review_dir)/'agent-review.md',delivery/'reports/agent-review.md')
    summary=('# Hasil rekonstruksi '+bundle['job_id']+'\n\nBuka [room tour](room_tour/index.html) di browser; HTML offline.\n\n'
      'Mesh detail: `mesh/`. Denah draft: `denah/`. Cakupan/review: `reports/`.\n\n'
      f'Review: {state}. Kualitas: {quality}. Skala: relatif; belum ada ukuran nyata.\n\n'
      'Komponen tetap terpisah. Denah irisan belum merupakan denah arsitektur final; furnitur dapat muncul sebagai garis. '
      'Kalibrasi ukuran pada HTML memerlukan jarak nyata dan tidak menjamin distorsi geometri telah tervalidasi.\n')
    with (delivery/'README.md').open('x') as stream:stream.write(summary)
    artifacts=[dict(file=str(p.relative_to(delivery)),sha256=sha256_file(p),size_bytes=p.stat().st_size)
               for p in sorted(delivery.rglob('*')) if p.is_file()]
    manifest=dict(schema_version=1,job_id=bundle['job_id'],execution_status='completed',review_status=state,
                  quality_status=quality,scale_status='relative',full_room_reconstruction_complete=False,
                  coverage=bundle['coverage'],failed_components=bundle.get('failed_components',[]),artifacts=artifacts)
    with (delivery/'manifest.json').open('x') as stream:json.dump(manifest,stream,indent=2,allow_nan=False)
    atomic_json(delivery.parent/'latest.json',dict(delivery=delivery.name,review_status=state,quality_status=quality))
    return manifest


def clone_delivery(root,source,job_id,*,delivery_root=None):
    """Attach a manual review to exactly the reviewed product bytes in a new version."""
    validate_job_id(job_id)
    root=Path(root).resolve();source=Path(source).resolve()
    if not source.is_relative_to(root/'end_product') or source.is_relative_to(root/'end_product/part1'):raise ValueError('Product source escapes allowed products')
    base=(workspace_path(root,delivery_root,must_exist=False) if delivery_root else source.parent.parent)/job_id
    if not base.resolve().is_relative_to(root/'end_product') or base.resolve().is_relative_to(root/'end_product/part1'):raise ValueError('Product target escapes allowed products')
    number=1
    while (base/f'v{number:03d}').exists():number+=1
    destination=base/f'v{number:03d}';destination.mkdir(parents=True)
    for path in sorted(source.rglob('*')):
        relative=path.relative_to(source)
        if not path.is_file() or str(relative) in ('README.md','manifest.json','reports/agent-review.json','reports/agent-review.md'):continue
        _copy(path,destination/relative)
    return destination
