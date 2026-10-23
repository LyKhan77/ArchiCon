"""Laptop controller: durable server worker, verified products and visual review."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

import yaml

from .pipeline_state import atomic_json,FileLock,workspace_path,WaitingResources,fingerprint
from .pipeline_layout import job_dir,create_job_dir,job_asset,is_nested_job
from .provenance import sha256_file


def controller_progress(root,job,phase,**details):
    """Durable UI phase; numerical attempt states remain owned by StageRunner."""
    path=job_dir(root,job['job_id'])/'controller-progress.json'
    try:previous=json.loads(path.read_text())
    except (OSError,ValueError):previous={}
    now=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    started=now if phase=='transfer' else previous.get('started_at',now)
    atomic_json(path,dict(phase=phase,started_at=started,updated_at=now,**details))


def register_job(root,video,job_id,machine_config,pipeline_config,*,ssh_target=None,reuse_config=None,review_mode='manual'):
    root=Path(root).resolve()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,31}',job_id):raise ValueError('job-id must be 1–32 letters/digits/_/-')
    video=Path(video);video=(root/video).resolve() if not video.is_absolute() else video.resolve()
    if not video.is_file():raise ValueError('Input video missing')
    machine=yaml.safe_load(workspace_path(root,machine_config).read_text())
    profile=yaml.safe_load(workspace_path(root,pipeline_config).read_text())
    if ssh_target and not re.fullmatch(r'[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+',ssh_target):raise ValueError('Unsafe SSH target')
    digest=sha256_file(video)
    basename=re.sub(r'[^A-Za-z0-9_.-]','_',video.name).lower()
    local=str(video.relative_to(root)) if video.is_relative_to(root) else str(video)
    directory=create_job_dir(root,job_id)
    server_root=machine['paths']['project_root'] if ssh_target else str(root)
    machine['paths']['project_root']=server_root
    machine_path=job_asset(root,directory,'config','machine.yaml');machine_path.parent.mkdir(parents=True,exist_ok=True);machine_path.write_text(yaml.safe_dump(machine,sort_keys=False))
    processing=dict(profile.get('processing',{}))
    processing['target_fps']=processing.pop('keyframe_target_fps',processing.get('target_fps',2))
    job=dict(schema_version=1,layout_version=2,job_id=job_id,title=video.stem+' · ruang dan lintasan',
      local_input_video=local,input_video='raw_videos/'+digest[:12]+'_'+basename,input_sha256=digest,
      machine_config=str(machine_path.relative_to(root)),processing=processing,review=dict(profile.get('review',{'required':True})),
      delivery_root=profile.get('delivery',{}).get('root','end_product'),
      transport=dict(mode='ssh' if ssh_target else 'local',ssh_target=ssh_target,server_root=server_root))
    if reuse_config:job['reuse']=json.loads(workspace_path(root,reuse_config).read_text())
    if review_mode!='manual':raise ValueError('Only manual review is enabled')
    job['review'].update(mode='manual',provider='manual')
    atomic_json(directory/'job.json',job)
    atomic_json(directory/'job-status.json',dict(execution_status='queued',review_status='pending',quality_status='unassessed',scale_status='relative'))
    return job


def describe_plan(job):
    return dict(job_id=job['job_id'],input_sha256=job['input_sha256'],transport=job['transport'],
      S00='Verify input and transfer',S01='Pinned runtime/resource guards',S02='Decode full timeline and audit',
      S03='Camera components and trajectory checks',S04='Frame/grid/hash mask manifest',S05='Bounded DA3',
      S06='Filtering + TSDF',S06R='Camera renders and orientation provenance',S07='Plan + viewer mesh',
      S08='Standalone HTML + browser QA',S09='Manual visual review handoff',S10='Bounded typed refinement after apply-review',S11='Verified delivery',
      reuse=job.get('reuse',{}),delivery_root=job.get('delivery_root','end_product/part2'))


def _command(argv,**kwargs):
    return subprocess.run(argv,check=True,text=True,**kwargs)


def remote_call(job,code,*arguments):
    transport=job['transport'];remote=transport['server_root']
    code="import sys;sys.path.insert(0,sys.argv[1]+'/src')\nfrom room_reconstruction.pipeline_layout import job_dir,job_asset\n"+code
    argv=[str(Path(remote)/'.venv/bin/python'),'-c',code,remote,*map(str,arguments)]
    if transport['mode']=='ssh':argv=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',transport['ssh_target'],shlex.join(argv)]
    return _command(argv,capture_output=True).stdout.strip()


def _sync(root,job,paths=None):
    if job['transport']['mode']!='ssh':return
    remote=job['transport'];target=remote['ssh_target']+':'+remote['server_root']+'/'
    if paths is None:
        _command(['rsync','-az','--files-from='+str(root/'.sync-files'),str(root)+'/',target])
    else:
        name=job_asset(root,job_dir(root,job['job_id']),'logs','transfer-list.txt');name.parent.mkdir(parents=True,exist_ok=True)
        name.write_text('\n'.join(paths)+'\n')
        _command(['rsync','-az','--files-from='+str(name),str(root)+'/',target])


def prepare_server(root,job):
    root=Path(root);directory=job_dir(root,job['job_id'])
    local=Path(job['local_input_video']);local=local if local.is_absolute() else root/local
    if sha256_file(local)!=job['input_sha256']:raise ValueError('Local video changed since registration')
    _sync(root,job)
    _sync(root,job,[str((directory/'job.json').relative_to(root)),job['machine_config']])
    # Reuse a byte-identical server copy if present; otherwise upload the selected input.
    code='''import sys,pathlib,hashlib,shutil,json
root=pathlib.Path(sys.argv[1]);dst=root/sys.argv[2];source=root/sys.argv[3];digest=sys.argv[4]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
dst.parent.mkdir(parents=True,exist_ok=True)
if not dst.exists() and source.is_file() and sha(source)==digest:shutil.copyfile(source,dst)
print(json.dumps({'ready':dst.is_file() and sha(dst)==digest,'exists':dst.exists()}))
'''
    result=json.loads(remote_call(job,code,job['input_video'],job['local_input_video'],job['input_sha256']))
    if result['exists'] and not result['ready']:raise ValueError('Server canonical video checksum differs')
    if not result['ready']:
        if job['transport']['mode']=='local':raise ValueError('Local canonical media preparation failed')
        remote=job['transport'];_command(['rsync','-az',str(local),remote['ssh_target']+':'+remote['server_root']+'/'+job['input_video']])
        if not json.loads(remote_call(job,code,job['input_video'],job['local_input_video'],job['input_sha256']))['ready']:
            raise ValueError('Uploaded video checksum differs')


def start_worker(job):
    code='''import sys,pathlib,subprocess,json
root=pathlib.Path(sys.argv[1]);jid=sys.argv[2];directory=job_dir(root,jid)
logfile=job_asset(root,directory,'logs','launch.log');logfile.parent.mkdir(parents=True,exist_ok=True)
with logfile.open('a') as log:
 p=subprocess.Popen([str(root/'.venv/bin/python'),str(root/'scripts/pipeline_worker.py'),'--job-id',jid],cwd=root,
  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'worker_pid':p.pid}))
'''
    return json.loads(remote_call(job,code,job['job_id']))


def worker_state(job):
    code='''import sys,pathlib,json
p=job_dir(pathlib.Path(sys.argv[1]),sys.argv[2])/'worker-status.json'
print(p.read_text() if p.exists() else json.dumps({'execution_status':'queued'}))
'''
    return json.loads(remote_call(job,code,job['job_id']))


def obtain_bundle(root,job):
    root=Path(root);directory=job_dir(root,job['job_id'])
    code="import sys,pathlib;print((job_dir(pathlib.Path(sys.argv[1]),sys.argv[2])/'bundle.json').read_text())"
    bundle=json.loads(remote_call(job,code,job['job_id']))
    if bundle['job_id']!=job['job_id'] or bundle['input_sha256']!=job['input_sha256']:raise ValueError('Remote bundle identity differs')
    for artifact in bundle['artifacts']:
        path=Path(artifact['file'])
        if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0]!='outputs':raise ValueError('Invalid transfer artifact path')
    if job['transport']['mode']=='ssh':
        listing=job_asset(root,directory,'logs','download-list.txt');listing.parent.mkdir(parents=True,exist_ok=True);listing.write_text('\n'.join(a['file'] for a in bundle['artifacts'])+'\n')
        transport=job['transport'];_command(['rsync','-az','--files-from='+str(listing),
              transport['ssh_target']+':'+transport['server_root']+'/',str(root)+'/'])
    from .pipeline_delivery import verify_bundle
    verify_bundle(root,bundle,input_sha256=job['input_sha256']);atomic_json(directory/'bundle.json',bundle)
    return bundle


def process_server(root,job,*,detach=False):
    controller_progress(root,job,'transfer')
    prepare_server(root,job)
    controller_progress(root,job,'processing')
    status=worker_state(job)
    if status['execution_status']=='complete':
        controller_progress(root,job,'downloading')
        return obtain_bundle(root,job)
    if status['execution_status']!='running':
        start_worker(job);time.sleep(.5)
    if detach:return None
    previous=None
    while True:
        status=worker_state(job);phase=status['execution_status']
        if phase!=previous:print('Server:',phase,flush=True);previous=phase
        if phase=='complete':
            controller_progress(root,job,'downloading')
            return obtain_bundle(root,job)
        if phase=='waiting_resources':
            controller_progress(root,job,'waiting_resources',reason=status.get('reason',''))
            atomic_json(job_dir(root,job['job_id'])/'job-status.json',dict(execution_status='waiting_resources',review_status='pending',quality_status='unassessed',scale_status='relative',reason=status.get('reason','')))
            raise WaitingResources('Waiting resources; resume later: '+status.get('reason',''))
        if phase=='failed':raise RuntimeError('Server stage failed: '+status.get('reason',''))
        time.sleep(2)


def browser_qa(root,job_dir,delivery):
    """Use pinned browser CLI against file:// with networking disabled."""
    root=Path(root);directory=Path(job_dir)/('qa' if is_nested_job(root,job_dir) else 'browser-qa')/delivery.name;directory.mkdir(parents=True,exist_ok=True)
    config=directory/'config.json'
    atomic_json(config,dict(allowUnrestrictedFileAccess=True,outputDir=str(directory),browser=dict(contextOptions=dict(viewport=dict(width=1440,height=1000)))))
    session=job_dir.name[:18]+'_'+fingerprint(job_dir.name)[:6] if is_nested_job(root,job_dir) else job_dir.name[:24]
    cli=['npm','exec','--yes','--package','@playwright/cli@0.1.22','--','playwright-cli','-s=qa_'+session]
    def call(*args):return _command(cli+list(args),cwd=root,capture_output=True,timeout=120).stdout
    try:
        call('open','--config',str(config),(delivery/'room_tour/index.html').as_uri())
        call('network-state-set','offline');call('reload')
        script='''async page=>{
 const errors=[];page.on("pageerror",e=>errors.push(e.message));await page.waitForFunction(()=>window.__TOUR_READY__);
 await page.screenshot({path:DEST+"/overview.png"});
 const count=await page.locator("#component option").count();
 const buttons=page.locator("#component-list [data-component]");
 const choose=async i=>{if(await buttons.count())await buttons.nth(i).click();else await page.locator("#component").selectOption({index:i});await page.waitForFunction(()=>document.getElementById("loading").hidden);};
 for(let i=0;i<count;i++)await choose(i);
 await choose(0);await page.locator("#tab-plan").click();
 await page.screenshot({path:DEST+"/floorplan.png"});
 await page.setViewportSize({width:390,height:844});
 const mobile=await page.evaluate(()=>document.documentElement.scrollWidth<=390);
 await page.screenshot({path:DEST+"/mobile.png"});
 if(!mobile||errors.length||await page.locator("#error").isVisible())throw new Error("Browser QA failed:"+errors.join(","));
 return {passed:true,offline:!(await page.evaluate(()=>navigator.onLine)),components:count,mobile_no_overflow:mobile,errors};
}'''.replace('DEST',json.dumps(str(directory)))
        stdout=call('run-code',script)
        match=re.search(r'### Result\s*\n([^\n]+)',stdout)
        if not match:raise RuntimeError('Browser CLI returned no QA result')
        result=json.loads(match.group(1));atomic_json(directory/'report.json',result)
        return result,[str(p.relative_to(root)) for p in directory.glob('*.png')]
    finally:
        try:call('close')
        except subprocess.SubprocessError:pass


def complete_job(root,job,bundle):
    from .pipeline_delivery import make_delivery,review_packet,finalize_delivery
    from .pipeline_review import prepare_manual_review
    root=Path(root).resolve();directory=job_dir(root,job['job_id']);state_path=directory/'job-status.json'
    controller_progress(root,job,'packaging')
    delivery=make_delivery(root,directory,bundle,delivery_root=job['delivery_root'])
    controller_progress(root,job,'browser_qa')
    qa,qa_images=browser_qa(root,directory,delivery)
    bundle=dict(bundle,qa=list(dict.fromkeys([*bundle.get('qa',[]),*qa_images])))
    atomic_json(directory/'review-bundle.json',bundle)
    manifest=finalize_delivery(root,delivery,bundle,None,None,pending=True)
    packet=review_packet(root,directory,bundle,delivery)
    packet['baseline_bundle_file']=str((directory/'review-bundle.json').relative_to(root))
    review_input=prepare_manual_review(root,directory,packet)
    atomic_json(state_path,dict(manifest,delivery=str(delivery.relative_to(root)),browser_qa=qa,
                               review_input=str(review_input.relative_to(root))))
    controller_progress(root,job,'awaiting_review')
    print(json.dumps(dict(job_id=job['job_id'],delivery=str(delivery.relative_to(root)),review_status='pending',
                          review_input=str(review_input.relative_to(root))),indent=2))
    return 0


def apply_manual_review(root,job,result_file,*,review_input=None):
    from .pipeline_delivery import clone_delivery,make_delivery,finalize_delivery,review_packet
    from .pipeline_review import load_manual_review,prepare_manual_review,refinement_presets
    root=Path(root).resolve();directory=job_dir(root,job['job_id']);state_path=directory/'job-status.json'
    state=json.loads(state_path.read_text());input_file=workspace_path(root,review_input or state['review_input'])
    packet=json.loads(input_file.read_text());result=load_manual_review(root,input_file,result_file)
    baseline=json.loads(workspace_path(root,packet['baseline_bundle_file']).read_text())
    chosen=packet.get('candidates',{}).get(result['selected_candidate_id'])
    if chosen:
        baseline=json.loads(workspace_path(root,chosen['bundle_file']).read_text());source=workspace_path(root,chosen['delivery'],must_exist=False)
    else:source=workspace_path(root,packet['delivery'],must_exist=False)
    review_directory=input_file.parent
    transaction_file=review_directory/'application.json'
    transaction=json.loads(transaction_file.read_text()) if transaction_file.exists() else {}
    saved_review=review_directory/'agent-review.json'
    if saved_review.exists():
        if json.loads(saved_review.read_text())!=result:raise ValueError('A different review was already applied to this request')
        if transaction.get('complete'):raise ValueError('This manual review was already applied')
    else:
        atomic_json(saved_review,result)
        (review_directory/'agent-review.md').write_text('# Manual agent review\n\n'+result['summary']+'\n')
    if result['decision']=='refine':
        state.update(execution_status='running',refinement_pending=dict(review_file=str(saved_review.relative_to(root)),review_input=str(input_file.relative_to(root))))
        atomic_json(state_path,state)
    product_bundle=dict(baseline,job_id=job['job_id'],producer_job_id=baseline['job_id'])
    if not transaction.get('state'):
        delivery=clone_delivery(root,source,job['job_id'],delivery_root=job['delivery_root'])
        manifest=finalize_delivery(root,delivery,product_bundle,result,review_directory)
        transaction['state']=dict(manifest,delivery=str(delivery.relative_to(root)),review_input=str(input_file.relative_to(root)))
        atomic_json(transaction_file,transaction)
    state=dict(transaction['state'])
    if result['decision']!='refine':
        atomic_json(state_path,state);transaction['complete']=True;atomic_json(transaction_file,transaction)
        print(json.dumps(state,indent=2));return 0
    state.update(execution_status='running',refinement_pending=dict(review_file=str(saved_review.relative_to(root)),review_input=str(input_file.relative_to(root))))
    atomic_json(state_path,state)
    round_number=packet['review_round']+1
    if round_number>min(2,int(job['processing'].get('max_refinement_rounds',2))):
        state.pop('refinement_pending',None)
        state.update(execution_status='completed',stop_reason='Refinement budget exhausted; manual decision/capture needed')
        atomic_json(state_path,state);transaction['complete']=True;atomic_json(transaction_file,transaction)
        print(state['stop_reason']);return 0
    candidates=dict(transaction.get('candidates',{}))
    presets=refinement_presets(result['refinement_requests'][0]['action'])[:min(2,int(job['processing'].get('max_candidates_per_round',2)))]
    for i,preset in enumerate(presets):
        if preset['id'] in candidates:
            # Reuse completed candidates only after rechecking bundle and product bytes.
            from .pipeline_delivery import verify_bundle
            cached=candidates[preset['id']]
            verify_bundle(root,json.loads(workspace_path(root,cached['bundle_file']).read_text()),input_sha256=job['input_sha256'])
            product=workspace_path(root,cached['delivery'],must_exist=False)
            for item in json.loads((product/'manifest.json').read_text())['artifacts']:
                if sha256_file(workspace_path(product,item['file']))!=item['sha256']:raise ValueError('Candidate delivery changed')
            continue
        cid=job['job_id'][:22]+f'_r{round_number}c{i+1}';candidate_dir=job_dir(root,cid) if job.get('layout_version')==2 else root/'outputs'/cid
        candidate=dict(job,job_id=cid,processing={**job['processing'],**preset['processing']},refinement_start=preset['start_stage'],
            reuse=dict(audit_run=baseline['audit_run'],camera_run=baseline['camera_run'],components=baseline['components'],tour_run=baseline['tour_run'],
                       failed_components=baseline.get('failed_components',[])))
        if not candidate_dir.exists():candidate_dir.mkdir(parents=True);atomic_json(candidate_dir/'job.json',candidate)
        else:candidate=json.loads((candidate_dir/'job.json').read_text())
        cb=process_server(root,candidate)
        cd=make_delivery(root,candidate_dir,cb,delivery_root=job['delivery_root'])
        _,images=browser_qa(root,candidate_dir,cd);cb=dict(cb,qa=list(dict.fromkeys([*cb['qa'],*images])))
        atomic_json(candidate_dir/'review-bundle.json',cb)
        finalize_delivery(root,cd,cb,None,None,pending=True)
        candidates[preset['id']]=dict(delivery=str(cd.relative_to(root)),bundle_file=str((candidate_dir/'review-bundle.json').relative_to(root)),
                                      evidence=cb['qa'],parameters=preset,component_evidence=cb.get('qa_component_evidence',{}),
                                      coverage=cb['coverage'],failed_components=cb.get('failed_components',[]))
        transaction['candidates']=candidates;atomic_json(transaction_file,transaction)
    next_packet=review_packet(root,directory,baseline,source,round_number,candidates)
    baseline_file=review_directory/'baseline-for-next-round.json';atomic_json(baseline_file,baseline)
    next_packet['baseline_bundle_file']=str(baseline_file.relative_to(root))
    next_packet['job_id']=job['job_id']
    next_packet['evidence']=list(dict.fromkeys([*next_packet['evidence'],*(p for c in candidates.values() for p in c['evidence'])]))
    next_input=prepare_manual_review(root,directory,next_packet)
    state.pop('refinement_pending',None)
    state.update(execution_status='completed',review_status='pending',quality_status='unassessed',review_input=str(next_input.relative_to(root)),review_round=round_number)
    atomic_json(state_path,state)
    transaction['complete']=True;atomic_json(transaction_file,transaction)
    print(json.dumps(dict(review_status='pending',review_input=str(next_input.relative_to(root)),candidates=list(candidates)),indent=2))
    return 0


def main(root,argv=None):
    root=Path(root).resolve();parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    run=sub.add_parser('run');run.add_argument('--input',required=True);run.add_argument('--job-id',required=True)
    run.add_argument('--machine-config',default='configs/paths.server.yaml');run.add_argument('--pipeline-config',default='configs/pipeline.yaml')
    run.add_argument('--ssh-target');run.add_argument('--local-worker',action='store_true');run.add_argument('--reuse-config');run.add_argument('--detach',action='store_true')
    run.add_argument('--profile',choices=['room-tour'],default='room-tour');run.add_argument('--review-agent',choices=['manual'],default='manual')
    for name in ('plan','status','resume','prepare-review','apply-review'):
        command=sub.add_parser(name);command.add_argument('--job-id',required=True)
        if name=='apply-review':command.add_argument('--review-file',required=True)
    args=parser.parse_args(argv)
    try:
        if args.action=='run':
            target=args.ssh_target
            if not args.local_worker and target is None:target=yaml.safe_load((root/'configs/paths.local.yaml').read_text())['server']['ssh_target']
            job=register_job(root,args.input,args.job_id,args.machine_config,args.pipeline_config,
                             ssh_target=None if args.local_worker else target,reuse_config=args.reuse_config,
                             review_mode='manual')
        else:job=json.loads(workspace_path(root,job_dir(root,args.job_id)/'job.json').read_text())
        directory=job_dir(root,job['job_id'])
        if args.action=='plan':print(json.dumps(describe_plan(job),indent=2));return 0
        if args.action=='status':
            state=json.loads((directory/'job-status.json').read_text());state['worker']=worker_state(job)
            print(json.dumps(state,indent=2));return 0
        with FileLock(directory/'controller.lock'):
            if args.action=='apply-review':return apply_manual_review(root,job,args.review_file)
            if args.action=='prepare-review':
                from .pipeline_review import prepare_manual_review
                current=json.loads((directory/'job-status.json').read_text())
                if current.get('refinement_pending'):raise ValueError('Resume pending refinement before preparing another review')
                packet=json.loads(workspace_path(root,current['review_input']).read_text())
                prepared=prepare_manual_review(root,directory,packet)
                current['review_input']=str(prepared.relative_to(root));atomic_json(directory/'job-status.json',current)
                print(str(prepared));return 0
            state_path=directory/'job-status.json'
            current=json.loads(state_path.read_text()) if state_path.exists() else {}
            if args.action=='resume' and current.get('refinement_pending'):
                pending=current['refinement_pending']
                return apply_manual_review(root,job,workspace_path(root,pending['review_file']),review_input=pending['review_input'])
            if args.action=='resume' and current.get('review_status') in ('pending','passed','needs_capture','needs_human_review') and current.get('execution_status')=='completed':
                print(json.dumps(current,indent=2));return 0
            atomic_json(state_path,dict(execution_status='running',review_status='pending',quality_status='unassessed',scale_status='relative'))
            bundle=process_server(root,job,detach=getattr(args,'detach',False))
            if bundle is None:print('Server worker started; use resume for packaging/review.');return 0
            return complete_job(root,job,bundle)
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
        print(str(exc),file=sys.stderr)
        if 'directory' in locals():
            atomic_json(directory/'controller-error.json',dict(reason=f'{type(exc).__name__}: {exc}'))
            state_file=directory/'job-status.json'
            state=json.loads(state_file.read_text()) if state_file.exists() else {}
            if state.get('execution_status') not in ('completed',):
                state.update(execution_status='waiting_resources' if isinstance(exc,WaitingResources) else 'failed',reason=str(exc))
                atomic_json(state_file,state)
        return 75 if isinstance(exc,WaitingResources) else 1
