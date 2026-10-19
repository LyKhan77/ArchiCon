"""Resumable stage attempts, atomic controller state and owned-process budgets."""
from __future__ import annotations

import ast
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

from .provenance import sha256_file
from .pipeline_layout import resolve_run_dir,reserve_run,run_id_exists,is_nested_job,job_asset


class WaitingResources(RuntimeError):
    pass


class StageExecutionError(RuntimeError):
    def __init__(self,message,run_dir,stage_status=None):
        super().__init__(message);self.run_dir=Path(run_dir);self.stage_status=stage_status or {}
        reason=str(self.stage_status.get('reason',''))
        infrastructure=('resource','memavailable','disk','gpu','runtime','backend','weights','snapshot','open3d','numpy','isolated','checksum','sha')
        self.quality_failure=(reason.startswith('ValueError:') or self.stage_status.get('error_type')=='ValueError') and not any(x in reason.lower() for x in infrastructure)
        self.resource_wait=any(x in reason.lower() for x in ('memavailable','disk free below','resource gate','gpu free memory','active compute processes','nonzero utilization'))


def atomic_json(path, data):
    path=Path(path);payload=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def fingerprint(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def workspace_path(root, path, *, must_exist=True):
    root=Path(root).resolve();path=Path(path)
    resolved=(root/path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to(root) or must_exist and not resolved.is_file():
        raise ValueError('Expected workspace file: '+str(path))
    return resolved


def input_signature(root,inputs):
    result={}
    for value in inputs:
        path=workspace_path(root,value,must_exist=False)
        if path.is_dir():
            members=sorted(path.rglob('*'))
            if any(p.is_symlink() for p in members):raise ValueError('Input directory symlinks are not cache-safe')
            result[str(path.relative_to(root))+'/']=fingerprint([str(p.relative_to(path)) for p in members])
            members=[p for p in members if p.is_file()]
        elif path.is_file():members=[path]
        else:raise ValueError('Missing stage input: '+str(value))
        for member in members:
            member=workspace_path(root,member)
            result[str(member.relative_to(root))]=sha256_file(member)
    return result


class FileLock:
    """Kernel-owned lock, automatically released after worker crashes."""
    def __init__(self,path):self.path=Path(path);self.stream=None
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True);self.stream=self.path.open('a+')
        try:fcntl.flock(self.stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close();self.stream=None
            raise RuntimeError('Already running: '+str(self.path))
        self.stream.seek(0);self.stream.truncate();self.stream.write(str(os.getpid()));self.stream.flush()
        return self
    def __exit__(self,*unused):
        if self.stream:
            fcntl.flock(self.stream,fcntl.LOCK_UN);self.stream.close();self.stream=None


def _processes():
    rows={}
    if Path('/proc').is_dir():
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():continue
            try:
                fields=(path/'stat').read_text().rsplit(')',1)[1].split()
                rows[int(path.name)]=(int(fields[1]),int(fields[21])*os.sysconf('SC_PAGE_SIZE'))
            except (OSError,ValueError,IndexError):continue
    else:
        result=subprocess.run(['ps','-axo','pid=,ppid=,rss='],capture_output=True,text=True,timeout=8,check=True)
        for line in result.stdout.splitlines():
            pid,parent,rss=map(int,line.split());rows[pid]=(parent,rss*1024)
    return rows


def _owned(pid,rows):
    result={pid}
    while True:
        expanded=result|{p for p,(parent,_) in rows.items() if parent in result}
        if expanded==result:return result
        result=expanded


def terminate_owned(proc):
    if proc.poll() is not None:return
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL);proc.wait()


def _gpu_initial(config,root):
    from .resources import ResourcePolicy,parse_gpu_csv,parse_compute_process_uuids,parse_mem_available,evaluate_resources
    def command(*args):return subprocess.run(['nvidia-smi',*args],capture_output=True,text=True,timeout=10,check=True).stdout
    policy=ResourcePolicy.from_mapping(config.get('compute',{}))
    try:
        gpus=parse_gpu_csv(command('--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'))
        busy=parse_compute_process_uuids(command('--query-compute-apps=gpu_uuid','--format=csv,noheader,nounits'))
        import shutil
        result=evaluate_resources(gpus=gpus,policy=policy,selector=config['compute']['gpu_selector'],
          mem_available_bytes=parse_mem_available(Path('/proc/meminfo').read_text()),disk_free_bytes=shutil.disk_usage(root).free,
          busy_uuids=busy)
    except (OSError,ValueError,subprocess.SubprocessError) as exc:raise WaitingResources(str(exc)) from exc
    if not result['resource_ready']:raise WaitingResources('; '.join(result['errors']))
    return result['selected_gpu']['uuid']


def _cpu_initial(config,root):
    if not Path('/proc/meminfo').is_file():return
    import shutil
    from .resources import ResourcePolicy,parse_mem_available,evaluate_resources
    policy=ResourcePolicy.from_mapping(config.get('compute',{}))
    result=evaluate_resources(gpus=[],policy=policy,selector=None,mem_available_bytes=parse_mem_available(Path('/proc/meminfo').read_text()),
                               disk_free_bytes=shutil.disk_usage(root).free,busy_uuids=None,gpu_required=False)
    if not result['resource_ready']:raise WaitingResources('; '.join(result['errors']))


def _gpu_usage(owned,uuid):
    result=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],
                          capture_output=True,text=True,check=True,timeout=10)
    total=0
    for row in csv.reader(result.stdout.splitlines()):
        if not row:continue
        if len(row)!=3:raise RuntimeError('GPU process telemetry malformed')
        device,pid,memory=(x.strip() for x in row)
        if int(pid) in owned:
            if device!=uuid:raise RuntimeError('Owned process allocated on unselected GPU')
            total+=int(memory)*1024**2
    return total


def _source_files(root,command):
    """Hash local import dependencies without importing CUDA/backend packages."""
    pending=[Path(__file__).resolve()]
    pending += [(root/arg).resolve() for arg in command if arg.endswith('.py') and (root/arg).is_file()]
    found={}
    while pending:
        path=pending.pop()
        if path in found or not path.is_relative_to(root):continue
        found[path]=sha256_file(path)
        try:tree=ast.parse(path.read_text())
        except (OSError,SyntaxError,UnicodeDecodeError):continue
        for node in ast.walk(tree):
            names=[]
            if isinstance(node,ast.Import):names=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:
                if node.level:
                    base=path.parent
                    for _ in range(node.level-1):base=base.parent
                    names=[];candidate=base.joinpath(*node.module.split('.')).with_suffix('.py')
                    if candidate.is_file():pending.append(candidate)
                else:names=[node.module]
            for name in names:
                for base in (root/'src',root/'scripts',root/'vendor'):
                    candidate=base.joinpath(*name.split('.')).with_suffix('.py')
                    if candidate.is_file():pending.append(candidate)
    return {str(p.relative_to(root)):h for p,h in sorted(found.items())}


class StageRunner:
    def __init__(self,root,job_dir,config):
        self.root=Path(root).resolve();self.job_dir=Path(job_dir).resolve();self.config=config
        if not self.job_dir.is_relative_to(self.root/'outputs'):raise ValueError('Job must stay under outputs')
        self.job_dir.mkdir(parents=True,exist_ok=True)
        self.state_path=self.job_dir/'stages.json'
        self.state=json.loads(self.state_path.read_text()) if self.state_path.exists() else {}

    def event(self,kind,**values):
        row=dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),event=kind,**values)
        with (self.job_dir/'events.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')

    def _save(self):atomic_json(self.state_path,self.state)

    def run(self,stage,inputs,params,command,*,gpu=False,timeout=3600,expected=()):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,55}',stage):raise ValueError('Unsafe stage name')
        if not command or not all(isinstance(x,str) for x in command) or timeout<=0:raise ValueError('Invalid stage command/timeout')
        files=input_signature(self.root,inputs)
        spec=dict(inputs=files,params=params,command=command,gpu=gpu,compute=self.config.get('compute',{}),
                  sources=_source_files(self.root,command),python=sys.version,expected=list(expected))
        identity=fingerprint(spec)
        records=self.state.setdefault(stage,[])
        for previous in reversed(records):
            if previous.get('fingerprint')!=identity or previous.get('status')!='passed':continue
            run=resolve_run_dir(self.root,previous['run_id'])
            if all((run/name).is_file() and sha256_file(run/name)==digest for name,digest in previous['artifacts'].items()):
                self.event('stage_reused',stage=stage,run_id=run.name);return run
        number=len(records)+1
        prefix=self.job_dir.name[:24]+'_'+stage[:30]+'_'+fingerprint(stage)[:8]
        if is_nested_job(self.root,self.job_dir):
            prefix=self.job_dir.name[:20]+'_'+fingerprint(self.job_dir.name)[:8]+'_'+stage[:28]+'_'+fingerprint(stage)[:8]
        while run_id_exists(self.root,f'{prefix}_A{number:03d}'):number+=1
        run_id=f'{prefix}_A{number:03d}';run=reserve_run(self.root,self.job_dir,stage,run_id)
        argv=[x.replace('@RUN_ID@',run_id) for x in command]
        record=dict(run_id=run_id,fingerprint=identity,status='running',spec=spec,command=argv,artifacts={})
        records.append(record);self._save();self.event('stage_started',stage=stage,run_id=run_id)
        env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
        env['PYTHONPATH']=str(self.root/'src')+os.pathsep+str(self.root/'vendor')
        compute=self.config.get('compute',{})
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]=str(min(4,int(compute.get('cpu_threads',4))))
        start=time.monotonic();proc=None;lock=None;peak=0;gpu_peak=0
        try:
            if not gpu:_cpu_initial(self.config,self.root)
            if gpu:
                uuid=str(compute.get('gpu_selector',''))
                if not uuid.startswith('GPU-'):raise ValueError('GPU requires explicit UUID selector')
                lock=FileLock(self.root/'outputs'/'.locks'/f'{uuid}.lock')
                try:lock.__enter__()
                except RuntimeError as exc:raise WaitingResources(str(exc)) from exc
                uuid=_gpu_initial(self.config,self.root);env['CUDA_VISIBLE_DEVICES']=uuid
            log_path=job_asset(self.root,self.job_dir,'logs',run_id+'.log');log_path.parent.mkdir(parents=True,exist_ok=True)
            with log_path.open('x') as log:
                proc=subprocess.Popen(argv,cwd=self.root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                while proc.poll() is None:
                    if time.monotonic()-start>timeout:raise RuntimeError('Stage timeout exceeded')
                    rows=_processes();owned=_owned(proc.pid,rows);rss=sum(rows.get(p,(0,0))[1] for p in owned);peak=max(peak,rss)
                    if rss>float(compute.get('target_host_rss_gib',6))*1024**3:raise RuntimeError('Owned process tree RSS budget exceeded')
                    if Path('/proc/meminfo').is_file():
                        from .resources import parse_mem_available
                        if parse_mem_available(Path('/proc/meminfo').read_text())<2*1024**3:raise WaitingResources('Host MemAvailable dropped below 2 GiB')
                    if gpu:
                        gpu_memory=_gpu_usage(owned,uuid);gpu_peak=max(gpu_peak,gpu_memory)
                        if gpu_memory>float(compute.get('target_process_gpu_gib',14))*1024**3:raise RuntimeError('Owned GPU memory budget exceeded')
                    time.sleep(min(1 if gpu else .1,max(.01,timeout/10)))
                if proc.wait()!=0:
                    status=json.loads((run/'status.json').read_text()) if (run/'status.json').is_file() else {}
                    error=StageExecutionError(f'Stage exited {proc.returncode}; see {run_id}.log',run,status)
                    if error.resource_wait:raise WaitingResources(str(status.get('reason',error)))
                    raise error
            state=json.loads((run/'status.json').read_text())
            if state.get('status') not in {'complete','completed','passed','camera_passed','depth_passed','mesh_passed','render_passed','exported','depth_complete'}:
                raise RuntimeError('Stage status is not successful: '+str(state.get('status')))
            for name in expected:workspace_path(run,name)
            if any(p.is_symlink() for p in run.rglob('*')):raise ValueError('Stage output symlink is not cache-safe')
            record['artifacts']={str(p.relative_to(run)):sha256_file(p) for p in sorted(run.rglob('*')) if p.is_file()}
            if input_signature(self.root,inputs)!=files:raise RuntimeError('Stage input changed during processing')
            record.update(status='passed',peak_rss_bytes=peak,peak_gpu_bytes=gpu_peak,elapsed_seconds=time.monotonic()-start)
            self._save();self.event('stage_passed',stage=stage,run_id=run_id);return run
        except BaseException as exc:
            if proc is not None:terminate_owned(proc)
            record.update(status='failed',reason=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start)
            self._save();self.event('stage_failed',stage=stage,run_id=run_id,reason=record['reason']);raise
        finally:
            if lock is not None:lock.__exit__(None,None,None)
