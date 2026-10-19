"""Manual review files and bounded, typed refinement requests."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .pipeline_state import atomic_json,workspace_path,fingerprint
from .provenance import sha256_file

DECISIONS=('accept','refine','needs_capture','needs_human_review')
QUALITIES=('unassessed','accepted','accepted_partial','needs_capture','rejected')
ACTIONS=('none','needs_capture','needs_measurement','needs_human_review','compare_filter_candidates','adjust_viewer_mesh')


def review_schema():
    def obj(properties):return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)
    string={'type':'string'}
    finding=obj(dict(id=string,severity=dict(type='string',enum=['minor','major','critical']),component_id=string,
        time_range_seconds=dict(type='array',items={'type':'number'},minItems=2,maxItems=2),
        evidence=dict(type='array',items=string),observation=string,hypothesis=string,
        confidence=dict(type='string',enum=['low','medium','high']),action=dict(type='string',enum=list(ACTIONS))))
    request=obj(dict(finding_id=string,start_stage=dict(type='string',enum=['S06','S07']),
                     action=dict(type='string',enum=['compare_filter_candidates','adjust_viewer_mesh'])))
    return obj(dict(schema_version=dict(type='integer',enum=[1]),job_id=string,review_round=dict(type='integer',minimum=0),review_input_sha256=string,
        decision=dict(type='string',enum=list(DECISIONS)),quality_status=dict(type='string',enum=list(QUALITIES)),summary=string,
        findings=dict(type='array',items=finding),refinement_requests=dict(type='array',items=request),
        required_user_actions=dict(type='array',items=string),scale_status=dict(type='string',enum=['relative','externally_calibrated_draft','validated']),
        full_room_reconstruction_complete={'type':'boolean'},selected_candidate_id=string))


def _schema_validate(value,schema,path='review'):
    kind=schema['type'];okay=False
    if kind=='object':okay=isinstance(value,dict)
    elif kind=='array':okay=isinstance(value,list)
    elif kind=='string':okay=isinstance(value,str)
    elif kind=='boolean':okay=type(value) is bool
    elif kind=='integer':okay=type(value) is int
    elif kind=='number':okay=type(value) in (int,float) and math.isfinite(value)
    if not okay:raise ValueError(f'{path}: expected {kind}')
    if 'enum' in schema and value not in schema['enum']:raise ValueError(f'{path}: unknown value')
    if kind=='object':
        if set(value)!=set(schema['required']):raise ValueError(f'{path}: missing/unknown fields')
        for key,child in schema['properties'].items():_schema_validate(value[key],child,path+'.'+key)
    if kind=='array':
        if len(value)<schema.get('minItems',0) or len(value)>schema.get('maxItems',10000):raise ValueError(path+': invalid length')
        for i,item in enumerate(value):_schema_validate(item,schema['items'],f'{path}[{i}]')
    if kind=='integer' and value<schema.get('minimum',value):raise ValueError(path+': too small')


def validate_review(value,*,job_id,component_ids,evidence,review_round=None,candidate_ids=('baseline',),scale_status='relative'):
    _schema_validate(value,review_schema())
    if value['job_id']!=job_id or review_round is not None and value['review_round']!=review_round:raise ValueError('Review job/round differs')
    if value['selected_candidate_id'] not in candidate_ids:raise ValueError('Unknown selected candidate')
    if value['scale_status']!=scale_status:raise ValueError('Reviewer cannot invent measured scale')
    if value['full_room_reconstruction_complete']:raise ValueError('Complete interiors require independent validation, not automatic review')
    findings={}
    for item in value['findings']:
        if item['id'] in findings or not item['id']:raise ValueError('Finding IDs must be unique')
        findings[item['id']]=item
        if item['component_id'] not in [*component_ids,'global']:raise ValueError('Finding component unknown')
        start,end=item['time_range_seconds']
        if start<0 or end<start:raise ValueError('Invalid evidence time range')
        if not item['evidence'] or not set(item['evidence']).issubset(set(evidence)):raise ValueError('Finding evidence unknown')
    for request in value['refinement_requests']:
        finding=findings.get(request['finding_id'])
        if finding is None or finding['action']!=request['action']:raise ValueError('Refinement must refer to matching finding')
        expected='S06' if request['action']=='compare_filter_candidates' else 'S07'
        if request['start_stage']!=expected:raise ValueError('Refinement stage/action mismatch')
    if value['decision']=='refine' and not value['refinement_requests']:raise ValueError('Refine decision needs typed request')
    if len({item['action'] for item in value['refinement_requests']})>1:raise ValueError('Use one refinement action per round; multiple findings may share it')
    if value['decision']!='refine' and value['refinement_requests']:raise ValueError('Non-refine decision cannot run actions')
    if value['decision']=='accept' and value['quality_status'] not in ('accepted','accepted_partial'):raise ValueError('Accept needs accepted quality')
    if value['decision']=='needs_capture' and value['quality_status']!='needs_capture':raise ValueError('Capture decision mismatch')
    if value['decision']=='needs_human_review' and value['quality_status']!='unassessed':raise ValueError('Unresolved human review cannot certify quality')
    if value['decision']=='refine' and value['quality_status'] not in ('unassessed','rejected'):raise ValueError('Pending refinement cannot certify quality')
    return value


def refinement_presets(action):
    if action=='compare_filter_candidates':
        return [dict(id='filter_stricter',start_stage='S06',processing=dict(confidence_quantile=.05,voxel_divisor=80,sdf_voxels=6)),
                dict(id='surface_coarser',start_stage='S06',processing=dict(confidence_quantile=.02,voxel_divisor=60,sdf_voxels=8))]
    if action=='adjust_viewer_mesh':
        return [dict(id='viewer_detail',start_stage='S07',processing=dict(viewer_voxel_divisor=650)),
                dict(id='viewer_light',start_stage='S07',processing=dict(viewer_voxel_divisor=300))]
    raise ValueError('Unsupported refinement action')


def prepare_manual_review(root,job_dir,packet):
    """Prepare files only. Never start an agent or send evidence to a service."""
    root=Path(root).resolve();job_dir=Path(job_dir).resolve();packet=dict(packet)
    packet['product_manifests']={}
    products=[packet.get('delivery'),*(x.get('delivery') for x in packet.get('candidates',{}).values())]
    for name in products:
        if name:
            manifest=workspace_path(root,name+'/manifest.json')
            packet['product_manifests'][str(manifest.relative_to(root))]=sha256_file(manifest)
    bundle_files=[packet.get('baseline_bundle_file'),*(x.get('bundle_file') for x in packet.get('candidates',{}).values())]
    packet['bundle_hashes']={name:sha256_file(workspace_path(root,name)) for name in bundle_files if name}
    packet['evidence_hashes']={name:sha256_file(workspace_path(root,name)) for name in packet['evidence']}
    packet['review_input_sha256']=fingerprint({k:v for k,v in packet.items() if k!='review_input_sha256'})
    parent=job_dir/'review'/f'round_{packet["review_round"]:02d}';parent.mkdir(parents=True,exist_ok=True)
    number=1
    while (parent/f'request_{number:03d}').exists():number+=1
    directory=parent/f'request_{number:03d}';directory.mkdir()
    atomic_json(directory/'review-input.json',packet);atomic_json(directory/'review-schema.json',review_schema())
    prompt=('Review manual hasil rekonstruksi video. Jangan menjalankan rekonstruksi atau mengubah source. '
      'Buka bukti visual yang tercantum: source/render, overview, denah dan browser QA. Periksa setiap komponen world terpisah. '
      'Pisahkan gejala dan hipotesis, sebutkan timestamp serta path bukti yang tepat. Skala masih relatif; jangan mengarang meter atau ruangan lengkap. '
      'Hole ringan ditoleransi; area yang tidak terekam membutuhkan capture, bukan geometri buatan. '
      'Bila perlu refinement, gunakan hanya compare_filter_candidates/S06 atau adjust_viewer_mesh/S07. '
      'Jika ada kandidat, pilih selected_candidate_id yang tersedia hanya jika lebih baik tanpa regresi material. '
      'Keluarkan satu JSON mengikuti review-schema.json dengan job_id, review_round dan review_input_sha256 persis dari review-input.json. '
      'Jangan memanggil agent tambahan atau layanan review otomatis. Simpan JSON agar pengguna dapat menjalankan apply-review.\n\n'
      f'Input: {directory.relative_to(root)}/review-input.json\nSchema: {directory.relative_to(root)}/review-schema.json\n')
    (directory/'review-prompt.md').write_text(prompt)
    return directory/'review-input.json'


def load_manual_review(root,input_file,result_file):
    root=Path(root).resolve();packet=json.loads(Path(input_file).read_text());value=json.loads(Path(result_file).read_text())
    expected=fingerprint({k:v for k,v in packet.items() if k!='review_input_sha256'})
    if packet['review_input_sha256']!=expected or value.get('review_input_sha256')!=expected:raise ValueError('Review input fingerprint differs')
    for name,digest in packet['evidence_hashes'].items():
        if sha256_file(workspace_path(root,name))!=digest:raise ValueError('Review evidence changed: '+name)
    for name,digest in packet.get('bundle_hashes',{}).items():
        if sha256_file(workspace_path(root,name))!=digest:raise ValueError('Review bundle changed: '+name)
    for name,digest in packet.get('product_manifests',{}).items():
        manifest=workspace_path(root,name)
        if sha256_file(manifest)!=digest:raise ValueError('Reviewed product manifest changed')
        for item in json.loads(manifest.read_text())['artifacts']:
            if sha256_file(workspace_path(manifest.parent,item['file']))!=item['sha256']:raise ValueError('Reviewed product bytes changed')
    validate_review(value,job_id=packet['job_id'],component_ids=packet['component_ids'],evidence=packet['evidence'],
                    review_round=packet['review_round'],candidate_ids=packet.get('candidate_ids',['baseline']),scale_status=packet.get('scale_status','relative'))
    if value['decision']=='accept':
        selected=packet.get('candidates',{}).get(value['selected_candidate_id'],packet)
        evidence=selected.get('component_evidence',{})
        if set(evidence)!=set(packet['component_ids']) or any(not x.get('complete') for x in evidence.values()):
            raise ValueError('Missing visual evidence cannot be accepted')
        if (selected.get('failed_components') or selected.get('coverage',{}).get('gaps')) and value['quality_status']=='accepted':
            raise ValueError('Known component/frame gaps require partial quality status')
    return value
