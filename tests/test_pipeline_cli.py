import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import yaml
from unittest.mock import patch

from room_reconstruction.pipeline_cli import register_job,describe_plan,complete_job,apply_manual_review,main
from room_reconstruction.pipeline_state import atomic_json,WaitingResources


class PipelineCLITests(unittest.TestCase):
    def test_registration_is_capture_specific_without_historical_frame_constants(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'configs').mkdir();(root/'raw_videos').mkdir()
            video=root/'raw_videos/new.MOV';video.write_bytes(b'new capture')
            config=root/'configs/machine.yaml';config.write_text(yaml.safe_dump(dict(profile='server',paths=dict(project_root='/server/project'),compute={})))
            profile=root/'configs/pipeline.yaml';profile.write_text(yaml.safe_dump(dict(processing={'target_fps':2},review={'required':True})))
            job=register_job(root,video,'fresh_job',config,profile,ssh_target='user@server')
            self.assertEqual(job['job_id'],'fresh_job');self.assertNotIn('302',json.dumps(job))
            self.assertIn('new.mov',job['input_video']);self.assertTrue(job['review']['required'])
            self.assertEqual(job['transport']['server_root'],'/server/project')
            self.assertIn('S09',describe_plan(job))
            self.assertEqual(job['layout_version'],2)
            self.assertTrue((root/'outputs/jobs/fresh_job/job.json').is_file())
            self.assertEqual(job['machine_config'],'outputs/jobs/fresh_job/config/machine.yaml')
            with self.assertRaises(FileExistsError):register_job(root,video,'fresh_job',config,profile,ssh_target='user@server')

    def test_unsafe_identity_rejected_before_outputs(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            with self.assertRaises(ValueError):register_job(root,root/'video','../escape',root/'machine',root/'profile',ssh_target='user@server')
            self.assertFalse((root/'outputs').exists())

    def test_processing_finishes_with_manual_review_pending_without_agent_call(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();directory=root/'outputs/job';directory.mkdir(parents=True)
            product=root/'end_product/part2/job/v001';product.mkdir(parents=True)
            image=root/'qa.png';image.write_bytes(b'QA evidence')
            job=dict(job_id='job',delivery_root='end_product/part2',review={'provider':'manual'},processing={})
            bundle=dict(job_id='job',components=[{'id':'main'}],coverage={'attempted':3,'processed':3,'gaps':[]},qa=['qa.png'])
            with patch('room_reconstruction.pipeline_delivery.make_delivery',return_value=product), \
                 patch('room_reconstruction.pipeline_cli.browser_qa',return_value=({'passed':True},[])), \
                 patch('subprocess.Popen',side_effect=AssertionError('Automatic agent prohibited')):
                self.assertEqual(complete_job(root,job,bundle),0)
            state=json.loads((directory/'job-status.json').read_text())
            self.assertEqual(state['review_status'],'pending')
            self.assertTrue((root/state['review_input']).is_file())
            progress=json.loads((directory/'controller-progress.json').read_text())
            self.assertEqual(progress['phase'],'awaiting_review')
            self.assertIn('updated_at',progress)

    def test_manual_refinement_resumes_after_resource_wait_without_reapplying_review(self):
        from room_reconstruction.pipeline_delivery import finalize_delivery,review_packet
        from room_reconstruction.pipeline_review import prepare_manual_review
        with TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();directory=root/'outputs/job';directory.mkdir(parents=True)
            product=root/'end_product/part2/job/v001';product.mkdir(parents=True)
            (product/'mesh.glb').write_bytes(b'reviewed geometry');(root/'qa.png').write_bytes(b'actual evidence')
            job=dict(job_id='job',delivery_root='end_product/part2',processing={'max_candidates_per_round':1},review={'provider':'manual'})
            atomic_json(directory/'job.json',job)
            bundle=dict(schema_version=1,job_id='job',input_sha256='a'*64,components=[{'id':'main'}],
                audit_run='audit',camera_run='camera',tour_run='tour',artifacts=[],qa=['qa.png'],
                qa_component_evidence={'main':{'complete':True}},coverage={'attempted':3,'processed':3,'gaps':[]})
            bundle['failed_components']=[dict(id='small_rejected_map',stage='S03_camera',reason='Too few observations')]
            atomic_json(directory/'review-bundle.json',bundle)
            finalize_delivery(root,product,bundle,None,None,pending=True)
            packet=review_packet(root,directory,bundle,product);packet['baseline_bundle_file']='outputs/job/review-bundle.json'
            location=prepare_manual_review(root,directory,packet);indexed=json.loads(location.read_text())
            atomic_json(directory/'job-status.json',dict(execution_status='completed',review_status='pending',review_input=str(location.relative_to(root))))
            result=dict(schema_version=1,job_id='job',review_round=0,review_input_sha256=indexed['review_input_sha256'],
                decision='refine',quality_status='unassessed',summary='Compare wall surfaces',
                findings=[dict(id='F1',severity='minor',component_id='main',time_range_seconds=[0,1],evidence=['qa.png'],
                    observation='Small wall holes',hypothesis='Sparse support',confidence='medium',action='compare_filter_candidates')],
                refinement_requests=[dict(finding_id='F1',start_stage='S06',action='compare_filter_candidates')],
                required_user_actions=[],scale_status='relative',full_room_reconstruction_complete=False,selected_candidate_id='baseline')
            result_file=directory/'manual.json';atomic_json(result_file,result)
            with patch('room_reconstruction.pipeline_cli.process_server',side_effect=WaitingResources('Insufficient RAM')):
                self.assertEqual(main(root,['apply-review','--job-id','job','--review-file',str(result_file)]),75)
            waiting=json.loads((directory/'job-status.json').read_text())
            self.assertEqual(waiting['execution_status'],'waiting_resources')
            self.assertIn('refinement_pending',waiting)
            reviewed_versions=list((root/'end_product/part2/job').glob('v*'))
            candidate_dir=root/'outputs/job_r1c1';candidate_product=root/'end_product/part2/job_r1c1/v001'
            candidate_product.mkdir(parents=True);(candidate_product/'mesh.glb').write_bytes(b'candidate geometry')
            candidate_bundle=dict(bundle,job_id='job_r1c1')
            with patch('room_reconstruction.pipeline_cli.process_server',return_value=candidate_bundle), \
                 patch('room_reconstruction.pipeline_delivery.make_delivery',return_value=candidate_product), \
                 patch('room_reconstruction.pipeline_cli.browser_qa',return_value=({'passed':True},[])), \
                 patch('subprocess.Popen',side_effect=AssertionError('Automatic agent prohibited')):
                self.assertEqual(main(root,['resume','--job-id','job']),0)
            state=json.loads((directory/'job-status.json').read_text())
            self.assertEqual(state['execution_status'],'completed');self.assertEqual(state['review_status'],'pending')
            self.assertNotIn('refinement_pending',state)
            self.assertEqual(len(list((root/'end_product/part2/job').glob('v*'))),len(reviewed_versions))
            next_packet=json.loads((root/state['review_input']).read_text())
            self.assertEqual(next_packet['review_round'],1);self.assertIn('filter_stricter',next_packet['candidates'])
            candidate_job=json.loads((candidate_dir/'job.json').read_text())
            self.assertEqual(candidate_job['reuse'].get('failed_components'),bundle['failed_components'])


if __name__=='__main__':unittest.main()
