"""Local job orchestration and bounded media transfer for ArchiCon."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile

import yaml

from .dashboard_snapshot import read_json, snapshot, tail
from .pipeline_cli import register_job
from .pipeline_layout import job_dir, validate_job_id
from .pipeline_state import atomic_json, FileLock
from .scope import normalize_scope

VIDEO_EXTENSIONS={'.mov','.mp4','.m4v','.webm','.avi','.mkv'}


class BusyError(RuntimeError):
    pass


class DashboardService:
    def __init__(self, root, *, python=None, controller=None, max_upload_bytes=20*1024**3):
        self.root=Path(root).resolve()
        self.python=str(python or self.root/'.venv/bin/python')
        self.controller=Path(controller or self.root/'scripts/dashboard_controller.py')
        self.max_upload_bytes=max_upload_bytes
        self.token=secrets.token_urlsafe(32)
        self.lock_path=self.root/'outputs/.locks/archicon-controller.lock'
        self.control_path=self.root/'outputs/.dashboard/control.json'
        self._children=[]; self._mutex=threading.Lock(); self._stop=threading.Event()

    def _safe(self, value, base):
        path=Path(value)
        if path.is_absolute() or '..' in path.parts: raise ValueError('Path tidak valid')
        target=self.root/path; allowed=self.root/base
        if not target.resolve().is_relative_to(allowed.resolve()): raise ValueError('Path di luar folder yang diizinkan')
        for parent in (target,*target.parents):
            if parent==self.root: break
            if parent.is_symlink(): raise ValueError('Symlink tidak didukung')
        return target

    def input_file(self, value):
        target=self._safe(value, 'raw_videos')
        if target.suffix.lower() not in VIDEO_EXTENSIONS or not target.is_file():
            raise ValueError('Pilih video yang tersedia di raw_videos')
        return target

    def inputs(self):
        result=[]
        for path in (self.root/'raw_videos').rglob('*'):
            if path.suffix.lower() not in VIDEO_EXTENSIONS or not path.is_file(): continue
            try: self.input_file(str(path.relative_to(self.root)))
            except ValueError: continue
            result.append(dict(path=str(path.relative_to(self.root)), name=path.name, size_bytes=path.stat().st_size))
        return sorted(result, key=lambda x:x['path'])

    def save_upload(self, reader, length, filename):
        extension=Path(filename).suffix.lower()
        if extension not in VIDEO_EXTENSIONS: raise ValueError('Format video tidak didukung')
        if not 0<length<=self.max_upload_bytes: raise ValueError('Ukuran upload di luar batas')
        if shutil.disk_usage(self.root).free<length+256*1024**2: raise ValueError('Ruang disk laptop tidak cukup untuk upload ini')
        base=self._safe('raw_videos/uploads/'+uuid.uuid4().hex, 'raw_videos')
        base.mkdir(parents=True, exist_ok=False); partial=base/'source.part'; target=base/('source'+extension)
        try:
            with partial.open('xb') as output:
                remaining=length
                while remaining:
                    block=reader.read(min(1024**2,remaining))
                    if not block: raise ValueError('Upload terputus; video belum disimpan')
                    output.write(block); remaining-=len(block)
                output.flush(); os.fsync(output.fileno())
            os.rename(partial,target)
            atomic_json(base/'upload.json',dict(original_name=Path(filename).name,size_bytes=length,completed_at=datetime.now(timezone.utc).isoformat()))
            return dict(path=str(target.relative_to(self.root)),name=Path(filename).name,size_bytes=length)
        except BaseException:
            shutil.rmtree(base); raise

    def _acquire(self):
        try: return FileLock(self.lock_path).__enter__()
        except RuntimeError as exc: raise BusyError('Masih ada job aktif. Tunggu hingga selesai atau menunggu resource.') from exc

    def control(self):
        with self._mutex:
            self._children=[p for p in self._children if p.poll() is None]
        try:
            lock=self._acquire(); lock.__exit__(None,None,None); busy=False
        except BusyError: busy=True
        return dict(read_json(self.control_path),busy=busy)

    def _launch(self, jid, lock):
        directory=job_dir(self.root,jid); log=directory/'logs/dashboard-controller.log';log.parent.mkdir(parents=True,exist_ok=True)
        fd=lock.stream.fileno()
        with log.open('ab') as output:
            proc=subprocess.Popen([self.python,str(self.controller),'--lock-fd',str(fd),'--job-id',jid],
                                  cwd=self.root,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
                                  start_new_session=True,pass_fds=(fd,))
        # Close only this descriptor: unlocking explicitly would release the child's inherited lock.
        lock.stream.close(); lock.stream=None
        with self._mutex: self._children.append(proc)
        atomic_json(self.control_path,dict(job_id=jid,pid=proc.pid,started_at=datetime.now(timezone.utc).isoformat()))
        def reap():
            code=proc.wait()
            if code:
                state_path=directory/'job-status.json';state=read_json(state_path)
                if state.get('execution_status') not in ('completed','failed','waiting_resources'):
                    state.update(execution_status='failed',reason='Controller berhenti; lihat log lalu gunakan Lanjutkan.')
                    atomic_json(state_path,state)
        threading.Thread(target=reap,name='archicon-controller-wait',daemon=True).start()
        return dict(job_id=jid,pid=proc.pid)

    def start_job(self, input_path, job_id, scope=None):
        validate_job_id(job_id); video=self.input_file(input_path); lock=self._acquire()
        try:
            target=yaml.safe_load((self.root/'configs/paths.local.yaml').read_text())['server']['ssh_target']
            if scope is not None:
                scope=normalize_scope(scope)
            if scope is not None and any(scope.get(key) is not None for key in ('start_time_seconds','end_time_seconds')):
                probe=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','json',str(video)],
                                     capture_output=True,text=True,check=True,timeout=60)
                duration=float(json.loads(probe.stdout)['format']['duration'])
                scope=normalize_scope(scope,duration_seconds=duration)
            register_job(self.root,video,job_id,'configs/paths.server.yaml','configs/pipeline.yaml',
                         ssh_target=target,review_mode='manual',scope=scope)
            return self._launch(job_id,lock)
        finally: lock.__exit__(None,None,None)

    def resume_job(self, jid):
        self._job(jid); state=read_json(job_dir(self.root,jid)/'job-status.json')
        if state.get('execution_status')=='completed': raise ValueError('Komputasi selesai; hasil menunggu review manual')
        lock=self._acquire()
        try: return self._launch(jid,lock)
        finally: lock.__exit__(None,None,None)

    def _job(self,jid):
        directory=job_dir(self.root,jid); job=read_json(directory/'job.json')
        if job.get('job_id')!=jid: raise FileNotFoundError('Job tidak ditemukan')
        return job

    def jobs(self):
        files=list((self.root/'outputs/jobs').glob('*/job.json'))+list((self.root/'outputs').glob('*/job.json'))
        result=[]
        for path in sorted(files,key=lambda p:p.stat().st_mtime,reverse=True):
            try:
                jid=path.parent.name; job=self._job(jid); state=read_json(path.parent/'job-status.json')
                result.append(dict(job_id=jid,title=job.get('title',jid),execution_status=state.get('execution_status','unknown'),
                                   review_status=state.get('review_status','pending'),quality_status=state.get('quality_status','unassessed')))
            except (OSError,ValueError,FileNotFoundError): continue
        return result

    def _public(self,jid):
        self._job(jid); directory=job_dir(self.root,jid); state=read_json(directory/'job-status.json'); public={}
        delivery=state.get('delivery')
        if delivery:
            base=self._safe(delivery,'end_product'); manifest=read_json(base/'manifest.json')
            for entry in manifest.get('artifacts',[]):
                rel=Path(entry['file'])
                if rel.is_absolute() or '..' in rel.parts: continue
                path=self._safe(str((base/rel).relative_to(self.root)),'end_product')
                if path.is_file(): public[str(path.relative_to(self.root))]='product'
            if (base/'manifest.json').is_file(): public[str((base/'manifest.json').relative_to(self.root))]='product'
        for path in (directory/'qa').rglob('*'):
            if path.suffix.lower() in ('.png','.jpg','.jpeg') and path.is_file():
                value=str(path.relative_to(self.root)); self._safe(value,'outputs'); public[value]='preview'
        for entry in read_json(directory/'dashboard-snapshot.json').get('previews',[]):
            try:
                value=entry['path']; path=self._safe(value,'outputs')
                # Only this job's nested attempts or its downloaded qa/live evidence.
                if path.is_relative_to(directory) and path.suffix.lower() in ('.png','.jpg','.jpeg') and path.is_file(): public[value]='preview'
            except (KeyError,ValueError): continue
        review=state.get('review_input')
        if review:
            path=self._safe(review,'outputs')
            if path.is_relative_to(directory/'review') and path.is_file(): public[review]='review'
        return public

    def artifact_file(self,jid,value):
        if value not in self._public(jid): raise ValueError('File bukan artefak publik job ini')
        base='end_product' if value.startswith('end_product/') else 'outputs'
        return self._safe(value,base)

    def product_zip(self,jid):
        state=read_json(job_dir(self.root,jid)/'job-status.json')
        if not state.get('delivery'): raise ValueError('Produk belum tersedia')
        base=self._safe(state['delivery'],'end_product')
        folder=self.root/'outputs/.dashboard/downloads';folder.mkdir(parents=True,exist_ok=True)
        fd,name=tempfile.mkstemp(suffix='.zip',dir=folder);os.close(fd);archive=Path(name)
        try:
            with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
                for value,kind in self._public(jid).items():
                    if kind=='product':
                        path=self.artifact_file(jid,value);z.write(path,path.relative_to(base))
            return archive
        except BaseException: archive.unlink(missing_ok=True);raise

    def review_session(self,jid):
        """Return a safe, global review summary for the completed job."""
        job=self._job(jid);directory=job_dir(self.root,jid);state=read_json(directory/'job-status.json')
        result=dict(available=False,applied=state.get('review_status') not in (None,'pending'),
                    review_input=state.get('review_input'),coverage=state.get('coverage') or {},
                    component_count=0,evidence=[],delivery=state.get('delivery'),
                    quality_status=state.get('quality_status','unassessed'))
        review_input=state.get('review_input')
        if not review_input:return result
        path=self._safe(review_input,'outputs')
        if not path.is_relative_to(directory/'review') or not path.is_file():return result
        packet=read_json(path)
        public=self._public(jid)
        def evidence_rank(item):
            name,kind=item;lower=name.lower();suffix=Path(name).suffix.lower()
            if kind=='preview' and ('overview' in lower or 'floorplan' in lower):return 0
            if kind=='preview':return 1
            if '/denah/' in lower and suffix in ('.png','.svg'):return 2
            if suffix=='.html':return 3
            return 9
        evidence=[name for name,kind in sorted(public.items(),key=evidence_rank)
                  if kind=='preview' or Path(name).suffix.lower() in ('.png','.jpg','.jpeg','.svg','.html')]
        result.update(available=True,coverage=packet.get('coverage') or result['coverage'],
                      component_count=len(packet.get('component_ids',[])),evidence=evidence[:16])
        return result

    def apply_dashboard_review(self,jid,decision,reason=''):
        """Apply one simple global decision through the existing review contract."""
        mapping={'accept':('accept','accepted'),'needs_review':('needs_human_review','unassessed'),
                 'reject':('needs_capture','needs_capture')}
        if decision not in mapping:raise ValueError('Keputusan review tidak valid')
        reason=str(reason or '').strip()
        if decision=='reject' and not reason:raise ValueError('Alasan wajib untuk Reject')
        job=self._job(jid);directory=job_dir(self.root,jid);state=read_json(directory/'job-status.json')
        session=self.review_session(jid)
        if not session['available']:raise ValueError('Sesi review belum tersedia')
        input_file=self._safe(session['review_input'],'outputs');review_dir=input_file.parent
        existing=read_json(review_dir/'agent-review.json')
        if session['applied'] and existing:
            expected=mapping[decision][0]
            if existing.get('decision')==expected:
                return dict(existing,delivery=state.get('delivery'),review_file=str((review_dir/'agent-review.json').relative_to(self.root)),already_applied=True)
            raise ValueError('Review untuk sesi ini sudah diterapkan')
        packet=read_json(input_file);coverage=packet.get('coverage') or {}
        internal,quality=mapping[decision]
        if decision=='accept' and (coverage.get('gaps') or packet.get('failed_components')):quality='accepted_partial'
        summary=reason or {'accept':'Manual review diterima.','needs_review':'Menunggu review lanjutan.'}[decision]
        candidates=packet.get('candidate_ids') or ['baseline'];selected='baseline' if 'baseline' in candidates else candidates[0]
        payload=dict(schema_version=1,job_id=jid,review_round=packet['review_round'],
                     review_input_sha256=packet['review_input_sha256'],decision=internal,
                     quality_status=quality,summary=summary,findings=[],refinement_requests=[],
                     required_user_actions=[],scale_status=packet.get('scale_status','relative'),
                     full_room_reconstruction_complete=False,selected_candidate_id=selected)
        atomic_json(review_dir/'manual.json',payload)
        from .pipeline_cli import apply_manual_review
        apply_manual_review(self.root,job,str((review_dir/'manual.json').relative_to(self.root)),
                            review_input=str(input_file.relative_to(self.root)))
        updated=read_json(directory/'job-status.json')
        return dict(payload,delivery=updated.get('delivery'),review_file=str((review_dir/'agent-review.json').relative_to(self.root)))

    def job_view(self,jid):
        job=self._job(jid); directory=job_dir(self.root,jid); state=read_json(directory/'job-status.json')
        saved=read_json(directory/'dashboard-snapshot.json')
        live=saved or snapshot(self.root,jid)
        control=self.control(); active=control.get('busy') and control.get('job_id')==jid
        local_phase=read_json(directory/'controller-progress.json')
        status=state.get('execution_status','unknown')
        if active and status not in ('completed','waiting_resources'): status='running'
        public=self._public(jid)
        downloads=[dict(path=p,name=Path(p).name,kind=k) for p,k in public.items() if k!='preview']
        previews=[dict(path=p,label=Path(p).stem.replace('_',' ')) for p,k in public.items() if k=='preview']
        try:
            age=max(0,time.time()-datetime.fromisoformat(live['updated_at']).timestamp())
        except (KeyError,ValueError): age=None
        elapsed=sum(x.get('elapsed_seconds') or 0 for x in live.get('stages',[])) or None
        elapsed_basis='stage_sum'
        if local_phase.get('started_at'):
            try:
                start=datetime.fromisoformat(local_phase['started_at']).timestamp()
                end=datetime.fromisoformat(local_phase['updated_at']).timestamp() if status=='completed' else time.time()
                elapsed=max(0,end-start);elapsed_basis='controller'
            except (KeyError,ValueError):pass
        return dict(job_id=jid,title=job.get('title',jid),execution_status=status,review_status=state.get('review_status','pending'),
                    quality_status=state.get('quality_status','unassessed'),scale_status=state.get('scale_status','relative'),
                    scope=job.get('scope',{'kind':'full_traversal','name':None,'start_time_seconds':None,'end_time_seconds':None,'source_policy':'full_input_video'}),
                    full_room_reconstruction_complete=state.get('full_room_reconstruction_complete',False),
                    coverage=state.get('coverage') or live.get('coverage'),failed_components=state.get('failed_components',[]),
                    reason=state.get('reason') or live.get('worker',{}).get('reason',''),phase=local_phase,
                    stages=live.get('stages',[]),worker=live.get('worker',{}),previews=previews[:12],downloads=downloads,
                    delivery=state.get('delivery'),input_path=job.get('local_input_video'),
                    review=self.review_session(jid),
                    elapsed_seconds=elapsed,elapsed_basis=elapsed_basis,
                    controller_active=bool(active),started_at=control.get('started_at') if active else None,
                    telemetry=dict(updated_at=live.get('updated_at'),stale=bool(active and (age is None or age>20)),error=live.get('error','')),
                    log=tail(directory/'logs/dashboard-controller.log',8000)+'\n'+str(live.get('log',''))[-16000:])

    def refresh(self,jid):
        job=self._job(jid);directory=job_dir(self.root,jid); transport=job.get('transport',{})
        try:
            if transport.get('mode')=='local': live=snapshot(self.root,jid)
            else:
                remote=transport['server_root']
                code='import sys,json;sys.path.insert(0,sys.argv[1]+"/src");from room_reconstruction.dashboard_snapshot import snapshot;print(json.dumps(snapshot(sys.argv[1],sys.argv[2])))'
                cmd=shlex.join([remote+'/.venv/bin/python','-c',code,remote,jid])
                response=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',transport['ssh_target'],cmd],capture_output=True,text=True,check=True,timeout=15)
                live=json.loads(response.stdout)
                for image in live.get('previews',[]):
                    rel=Path(image['path']);expected=Path('outputs/jobs')/jid
                    if rel.is_absolute() or '..' in rel.parts or not rel.is_relative_to(expected) or rel.suffix.lower() not in ('.png','.jpg','.jpeg'): continue
                    # Preserve the remote path below qa/live, avoiding filename collisions.
                    dest=directory/'qa/live'/rel.relative_to(expected);dest.parent.mkdir(parents=True,exist_ok=True)
                    if not dest.exists():
                        subprocess.run(['rsync','-az',transport['ssh_target']+':'+remote+'/'+str(rel),str(dest)],check=True,capture_output=True,timeout=20)
                    image['path']=str(dest.relative_to(self.root))
            live.pop('error',None);atomic_json(directory/'dashboard-snapshot.json',live)
        except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:
            live=read_json(directory/'dashboard-snapshot.json');live['error']='Progress server belum dapat diambil: '+str(exc)[:500]
            atomic_json(directory/'dashboard-snapshot.json',live)

    def start_monitor(self):
        def monitor():
            last_active=None
            while not self._stop.is_set():
                control=self.control();jid=control.get('job_id')
                if control.get('busy') and jid:
                    try: self.refresh(jid)
                    except (OSError,ValueError): pass
                    last_active=jid
                elif last_active:
                    # Keep the final stage/preview snapshot when the controller exits between polls.
                    try: self.refresh(last_active)
                    except (OSError,ValueError): pass
                    last_active=None
                self._stop.wait(3)
        thread=threading.Thread(target=monitor,name='archicon-progress',daemon=True);thread.start()

    def stop_monitor(self):
        self._stop.set()
