"""User-visible dashboard contracts; no GPU computation or external service."""
import http.client
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from room_reconstruction.dashboard_service import DashboardService, BusyError
from room_reconstruction.dashboard_http import make_server
from room_reconstruction.dashboard_snapshot import snapshot
from room_reconstruction.pipeline_state import atomic_json


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        (self.root/'configs').mkdir();(self.root/'raw_videos').mkdir()
        (self.root/'configs/paths.local.yaml').write_text('server:\n  ssh_target: user@host\n')
        (self.root/'configs/paths.server.yaml').write_text('profile: server\npaths:\n  project_root: /server/project\ncompute: {}\n')
        (self.root/'configs/pipeline.yaml').write_text('processing: {}\nreview:\n  required: true\ndelivery:\n  root: end_product\n')
        self.video=self.root/'raw_videos/source.MOV';self.video.write_bytes(b'video bytes')
        self.controller=self.root/'fake_controller.py'
        self.controller.write_text('''import os,sys,time,json
from pathlib import Path
fd=int(sys.argv[sys.argv.index('--lock-fd')+1]);os.fstat(fd)
jid=sys.argv[sys.argv.index('--job-id')+1]
p=Path('outputs/jobs')/jid/'job-status.json'
time.sleep(.7)
p.write_text(json.dumps({'execution_status':'completed','review_status':'pending','quality_status':'unassessed'}))
''')
        self.service=DashboardService(self.root,python=sys.executable,controller=self.controller,max_upload_bytes=1024)

    def job(self,jid='existing',legacy=False):
        p=self.root/('outputs' if legacy else 'outputs/jobs')/jid
        atomic_json(p/'job.json',{'job_id':jid,'title':'Interior','input_sha256':'abc','local_input_video':'raw_videos/source.MOV','transport':{'mode':'local','server_root':str(self.root)}})
        atomic_json(p/'job-status.json',{'execution_status':'completed','review_status':'pending','quality_status':'unassessed','scale_status':'relative','coverage':{'attempted':8,'processed':7,'gaps':['f8']}})
        return p

    def review_job(self, jid='existing'):
        directory=self.job(jid)
        review_dir=directory/'review/round_00/request_001';review_dir.mkdir(parents=True)
        packet={
            'schema_version':1,'job_id':jid,'scope':None,'review_round':0,
            'component_ids':['main'],'coverage':{'attempted':8,'processed':7,'gaps':['f8'],
                                                  'full_room_reconstruction_complete':False},
            'scale_status':'relative','failed_components':[],
            'evidence':['outputs/jobs/'+jid+'/qa/v001/overview.png'],
            'delivery':'end_product/'+jid+'/v001','candidate_ids':['baseline'],
            'candidates':{},'limitations':['Relative units; no measured reference.'],
            'baseline_bundle_file':'outputs/jobs/'+jid+'/review-bundle.json',
            'review_input_sha256':'fixture',
        }
        atomic_json(review_dir/'review-input.json',packet)
        state=json.loads((directory/'job-status.json').read_text())
        state['review_input']=str((review_dir/'review-input.json').relative_to(self.root))
        atomic_json(directory/'job-status.json',state)
        return directory,review_dir

    def test_review_session_exposes_safe_summary(self):
        directory,review_dir=self.review_job()
        qa=directory/'qa/v001';qa.mkdir(parents=True);(qa/'overview.png').write_bytes(b'preview')
        result=self.service.review_session('existing')
        self.assertTrue(result['available']);self.assertFalse(result['applied'])
        self.assertEqual(result['coverage']['gaps'],['f8'])
        self.assertEqual(result['component_count'],1)
        self.assertIn('outputs/jobs/existing/qa/v001/overview.png',result['evidence'])

    def test_dashboard_review_mapping_requires_reason_and_writes_schema_payload(self):
        directory,review_dir=self.review_job()
        with self.assertRaises(ValueError):self.service.apply_dashboard_review('existing','reject',' ')
        with patch('room_reconstruction.pipeline_cli.apply_manual_review',return_value=0) as apply:
            result=self.service.apply_dashboard_review('existing','reject','Geometry has large holes')
        self.assertEqual(result['decision'],'needs_capture');self.assertEqual(result['quality_status'],'needs_capture')
        payload=json.loads((review_dir/'manual.json').read_text())
        self.assertEqual(payload['decision'],'needs_capture');self.assertEqual(payload['summary'],'Geometry has large holes')
        self.assertFalse(payload['full_room_reconstruction_complete']);apply.assert_called_once()

    def test_dashboard_accept_with_gaps_maps_to_partial_quality(self):
        directory,review_dir=self.review_job()
        with patch('room_reconstruction.pipeline_cli.apply_manual_review',return_value=0):
            result=self.service.apply_dashboard_review('existing','accept','Usable for exploration')
        self.assertEqual(result['decision'],'accept');self.assertEqual(result['quality_status'],'accepted_partial')

    def test_upload_commits_only_complete_stream_and_lists_video(self):
        result=self.service.save_upload(io.BytesIO(b'123456'),6,'phone.MOV')
        self.assertEqual((self.root/result['path']).read_bytes(),b'123456')
        self.assertIn(result['path'],[v['path'] for v in self.service.inputs()])
        with self.assertRaises(ValueError):self.service.save_upload(io.BytesIO(b'12'),6,'phone.MOV')
        self.assertFalse(list((self.root/'raw_videos/uploads').rglob('*.part')))
        with self.assertRaises(ValueError):self.service.save_upload(io.BytesIO(b'x'),1025,'phone.MOV')
        with self.assertRaises(ValueError):self.service.save_upload(io.BytesIO(b'x'),1,'bad.html')

    def test_launch_lock_survives_new_service_and_releases_when_child_finishes(self):
        job=self.service.start_job('raw_videos/source.MOV','new_001')
        self.assertEqual(job['job_id'],'new_001')
        other=DashboardService(self.root,python=sys.executable,controller=self.controller)
        self.assertTrue(other.control()['busy'])
        with self.assertRaises(BusyError):other.start_job('raw_videos/source.MOV','new_002')
        self.assertFalse((self.root/'outputs/jobs/new_002').exists())
        deadline=time.monotonic()+5
        while other.control()['busy'] and time.monotonic()<deadline:time.sleep(.05)
        self.assertFalse(other.control()['busy'])
        self.assertEqual(other.job_view('new_001')['execution_status'],'completed')
        with self.assertRaises(FileExistsError):other.start_job('raw_videos/source.MOV','new_001')

    def test_unsafe_inputs_ids_and_symlinks_rejected(self):
        for path,jid in [('../outside.MOV','safe'),('raw_videos/source.MOV','../oops'),('raw_videos/source.MOV','part1')]:
            with self.assertRaises((ValueError,FileNotFoundError)):self.service.start_job(path,jid)
        (self.root/'raw_videos/linked.MOV').symlink_to(self.video)
        with self.assertRaises(ValueError):self.service.input_file('raw_videos/linked.MOV')

    def test_new_and_legacy_jobs_are_read_without_quality_acceptance(self):
        self.job();self.job('legacy',legacy=True)
        self.assertEqual({j['job_id'] for j in self.service.jobs()},{'existing','legacy'})

    def test_start_job_passes_single_room_scope_and_job_view_exposes_it(self):
        with patch('room_reconstruction.dashboard_service.register_job') as register:
            register.side_effect=lambda *args, **kwargs: atomic_json(
                self.root/'outputs/jobs/workshop_001/job.json',
                {'job_id':'workshop_001','scope':kwargs['scope']})
            with patch.object(self.service, '_launch', return_value={'job_id':'workshop_001'}):
                self.service.start_job('raw_videos/source.MOV','workshop_001',scope={
                    'kind':'single_room','name':'Workshop'})
            self.assertEqual(register.call_args.kwargs['scope']['kind'],'single_room')
            atomic_json(self.root/'outputs/jobs/workshop_001/job-status.json',
                        {'execution_status':'queued','review_status':'pending','quality_status':'unassessed'})
            view=self.service.job_view('workshop_001')
            self.assertEqual(view['scope']['name'],'Workshop')
    def test_snapshot_omits_huge_specs_bounds_logs_and_ignores_partial_event(self):
        p=self.job();atomic_json(p/'stages.json',{'S03_camera':[{'run_id':'r1','status':'passed','spec':{'big':'x'*10000},'artifacts':{'big':'hash'},'elapsed_seconds':2}]})
        (p/'logs').mkdir();(p/'logs/r1.log').write_text('x'*50000)
        (p/'events.jsonl').write_text('{incomplete')
        atomic_json(self.root/'outputs/r1/report.json',{'selected_frame_count':8,'accepted_union_count':7,'missing_frame_ids':['f8']})
        result=snapshot(self.root,'existing')
        self.assertEqual(result['stages'][0]['stage'],'S03_camera')
        self.assertNotIn('spec',json.dumps(result));self.assertLess(len(result['log']),25000)
        self.assertEqual(result['coverage']['processed'],7)

    def test_artifacts_allow_manifest_only_and_zip_preserves_bytes(self):
        p=self.job();delivery=self.root/'end_product/existing/v001';(delivery/'room_tour').mkdir(parents=True)
        (delivery/'room_tour/index.html').write_text('tour');(delivery/'private.txt').write_text('private')
        atomic_json(delivery/'manifest.json',{'artifacts':[{'file':'room_tour/index.html','sha256':'test','size_bytes':4}]})
        state=json.loads((p/'job-status.json').read_text());state['delivery']='end_product/existing/v001';atomic_json(p/'job-status.json',state)
        public='end_product/existing/v001/room_tour/index.html'
        self.assertEqual(self.service.artifact_file('existing',public).read_text(),'tour')
        for path in ['configs/paths.server.yaml','end_product/existing/v001/private.txt','../outside']:
            with self.assertRaises(ValueError):self.service.artifact_file('existing',path)
        import zipfile
        archive=self.service.product_zip('existing');self.addCleanup(archive.unlink,missing_ok=True)
        with zipfile.ZipFile(archive) as z:
            self.assertEqual(z.read('room_tour/index.html'),b'tour');self.assertNotIn('private.txt',z.namelist())

    def test_snapshot_failure_preserves_previous_progress_and_reports_error(self):
        import subprocess
        p=self.job();job=json.loads((p/'job.json').read_text())
        job['transport']={'mode':'ssh','ssh_target':'user@host','server_root':'/remote'};atomic_json(p/'job.json',job)
        atomic_json(p/'dashboard-snapshot.json',{'updated_at':'2026-01-01T00:00:00+00:00','stages':[{'stage':'S03_camera','status':'running'}]})
        with patch('room_reconstruction.dashboard_service.subprocess.run',side_effect=subprocess.TimeoutExpired('ssh',15)):
            self.service.refresh('existing')
        view=self.service.job_view('existing')
        self.assertEqual(view['stages'][0]['stage'],'S03_camera')
        self.assertIn('belum dapat diambil',view['telemetry']['error'])

    def test_completed_job_cannot_start_duplicate_controller(self):
        p=self.job()
        with self.assertRaises(ValueError):self.service.resume_job('existing')
        self.assertFalse(self.service.control()['busy'])
        atomic_json(p/'controller-progress.json',dict(phase='awaiting_review',started_at='2026-10-16T01:00:00Z',updated_at='2026-10-16T01:00:10Z'))
        view=self.service.job_view('existing')
        self.assertEqual(view['elapsed_seconds'],10)
        self.assertEqual(view['elapsed_basis'],'controller')

    def test_http_token_origin_host_and_video_range(self):
        server=make_server(self.service,port=0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        host=f'127.0.0.1:{server.server_port}'
        def request(method,path,body=None,headers=None):
            c=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=5)
            c.request(method,path,body=body,headers=headers or {});r=c.getresponse();data=r.read();status=r.status;h=dict(r.getheaders());c.close();return status,data,h
        status,body,_=request('GET','/api/config');self.assertEqual(status,200);token=json.loads(body)['token']
        self.assertEqual(request('POST','/api/jobs','{}',{'Content-Type':'application/json'})[0],403)
        self.assertEqual(request('GET','/api/config',headers={'Host':'attacker.invalid'})[0],403)
        headers={'X-ArchiCon-Token':token,'Origin':'https://attacker.invalid','Content-Type':'application/json'}
        self.assertEqual(request('POST','/api/jobs','{}',headers)[0],403)
        headers.update(Origin='http://'+host)
        self.assertEqual(request('POST','/api/jobs',json.dumps({'input_path':'raw_videos/source.MOV','job_id':'bad/id'}),headers)[0],400)
        with patch.object(self.service,'start_job',return_value={'job_id':'workshop_001'}) as start:
            status,body,_=request('POST','/api/jobs',json.dumps({'input_path':'raw_videos/source.MOV','job_id':'workshop_001',
                'scope':{'kind':'single_room','name':'Workshop'}}),headers)
            self.assertEqual(status,201)
            self.assertEqual(start.call_args.args[2]['kind'],'single_room')
        status,body,h=request('GET','/api/video?path=raw_videos/source.MOV',headers={'Range':'bytes=1-4'})
        self.assertEqual(status,206);self.assertEqual(body,b'ideo');self.assertEqual(h['Content-Range'],'bytes 1-4/11')
        self.assertEqual(request('GET','/api/video?path=../configs/paths.server.yaml')[0],400)
        headers.update({'Content-Type':'application/octet-stream'})
        status,body,_=request('POST','/api/uploads?filename=phone.MOV',b'upload bytes',headers)
        self.assertEqual(status,201)
        uploaded=json.loads(body)['path']
        self.assertEqual(request('GET','/api/video?path='+uploaded)[1],b'upload bytes')

    def test_http_manual_review_endpoint_requires_token_and_reject_reason(self):
        self.review_job();server=make_server(self.service,port=0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        host=f'127.0.0.1:{server.server_port}'
        def request(body,headers=None):
            c=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=5)
            c.request('POST','/api/jobs/existing/review',body=json.dumps(body),headers=headers or {'Content-Type':'application/json'});r=c.getresponse();data=r.read();status=r.status;c.close();return status,data
        status,_=request({'decision':'reject','reason':'x'});self.assertEqual(status,403)
        c=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=5);c.request('GET','/api/config');response=c.getresponse();token=json.loads(response.read())['token'];c.close()
        headers={'Content-Type':'application/json','X-ArchiCon-Token':token,'Origin':'http://'+host}
        self.assertEqual(request({'decision':'reject','reason':' '},headers)[0],400)
        with patch.object(self.service,'apply_dashboard_review',return_value={'decision':'needs_capture'}):
            status,body=request({'decision':'reject','reason':'holes'},headers)
        self.assertEqual(status,200);self.assertEqual(json.loads(body)['decision'],'needs_capture')


if __name__=='__main__':unittest.main()
