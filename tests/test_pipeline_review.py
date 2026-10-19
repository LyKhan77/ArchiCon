import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from room_reconstruction.pipeline_review import validate_review, refinement_presets,prepare_manual_review,load_manual_review


def review():
    return dict(schema_version=1,job_id='job',review_round=0,review_input_sha256='a'*64,decision='accept',quality_status='accepted_partial',
      summary='Observed coverage reviewed; relative scale.',findings=[],refinement_requests=[],required_user_actions=['Measure a reference'],
      scale_status='relative',full_room_reconstruction_complete=False,selected_candidate_id='baseline')


class ReviewTests(unittest.TestCase):
    def test_accept_partial_keeps_relative_units_and_actual_job_identity(self):
        value=review();validate_review(value,job_id='job',component_ids=['main'],evidence=['qa.png'])
        wrong=copy.deepcopy(value);wrong['job_id']='another'
        with self.assertRaises(ValueError):validate_review(wrong,job_id='job',component_ids=['main'],evidence=['qa.png'])

    def test_fabricated_evidence_and_unsupported_action_are_rejected(self):
        value=review();value['decision']='refine';value['quality_status']='unassessed'
        value['findings']=[dict(id='F1',severity='major',component_id='main',time_range_seconds=[0,1],evidence=['missing.png'],
          observation='Double wall',hypothesis='Depth mismatch',confidence='medium',action='compare_filter_candidates')]
        value['refinement_requests']=[dict(finding_id='F1',start_stage='S06',action='compare_filter_candidates')]
        with self.assertRaises(ValueError):validate_review(value,job_id='job',component_ids=['main'],evidence=['qa.png'])
        value['findings'][0]['evidence']=['qa.png'];validate_review(value,job_id='job',component_ids=['main'],evidence=['qa.png'])
        value['refinement_requests'][0]['action']='run_arbitrary_shell'
        with self.assertRaises(ValueError):validate_review(value,job_id='job',component_ids=['main'],evidence=['qa.png'])

    def test_review_cannot_invent_metric_scale_or_complete_rooms(self):
        value=review();value['scale_status']='validated'
        with self.assertRaises(ValueError):validate_review(value,job_id='job',component_ids=['main'],evidence=[])

    def test_pending_human_review_cannot_also_accept_quality(self):
        value=review();value['decision']='needs_human_review'
        with self.assertRaises(ValueError):validate_review(value,job_id='job',component_ids=['main'],evidence=[])
        value['quality_status']='unassessed';validate_review(value,job_id='job',component_ids=['main'],evidence=[])
        value=review();value['full_room_reconstruction_complete']=True
        with self.assertRaises(ValueError):validate_review(value,job_id='job',component_ids=['main'],evidence=[])

    def test_refinement_is_bounded_and_does_not_change_camera_world(self):
        candidates=refinement_presets('compare_filter_candidates')
        self.assertLessEqual(len(candidates),2)
        self.assertTrue(all(x['start_stage']=='S06' for x in candidates))
        self.assertTrue(all('camera' not in x['processing'] for x in candidates))
        with self.assertRaises(ValueError):refinement_presets('download_new_model')

    def test_manual_review_binds_actual_evidence_and_rejects_stale_image(self):
        import json
        with TemporaryDirectory() as temporary:
            root=Path(temporary);job=root/'outputs/job';job.mkdir(parents=True)
            image=root/'qa.png';image.write_bytes(b'actual review evidence')
            packet=dict(job_id='job',review_round=0,component_ids=['main'],component_evidence={'main':{'complete':True}},evidence=['qa.png'],scale_status='relative',candidate_ids=['baseline'])
            location=prepare_manual_review(root,job,packet)
            indexed=json.loads(location.read_text());value=review()
            value['review_input_sha256']=indexed['review_input_sha256']
            result=root/'result.json';result.write_text(json.dumps(value))
            self.assertEqual(load_manual_review(root,location,result)['decision'],'accept')
            image.write_bytes(b'changed evidence')
            with self.assertRaises(ValueError):load_manual_review(root,location,result)

    def test_selected_candidate_requires_its_own_complete_evidence(self):
        import json
        with TemporaryDirectory() as temporary:
            root=Path(temporary);directory=root/'outputs/job';directory.mkdir(parents=True)
            (root/'qa.png').write_bytes(b'actual evidence')
            packet=dict(job_id='job',review_round=0,component_ids=['main'],evidence=['qa.png'],
                component_evidence={'main':{'complete':True}},candidate_ids=['baseline','filtered'],
                candidates={'filtered':{'component_evidence':{'main':{'complete':False}},'coverage':{'gaps':[]}}})
            location=prepare_manual_review(root,directory,packet);value=review();value['selected_candidate_id']='filtered'
            value['review_input_sha256']=json.loads(location.read_text())['review_input_sha256']
            result=root/'review.json';result.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'Missing visual evidence'):load_manual_review(root,location,result)

    def test_review_bundle_cannot_change_after_manual_handoff(self):
        import json
        with TemporaryDirectory() as temporary:
            root=Path(temporary);directory=root/'outputs/job';directory.mkdir(parents=True)
            (root/'qa.png').write_bytes(b'actual evidence');bundle=directory/'bundle.json';bundle.write_text('{}')
            packet=dict(job_id='job',review_round=0,component_ids=['main'],evidence=['qa.png'],
                component_evidence={'main':{'complete':True}},baseline_bundle_file='outputs/job/bundle.json')
            location=prepare_manual_review(root,directory,packet);value=review()
            value['review_input_sha256']=json.loads(location.read_text())['review_input_sha256']
            result=root/'review.json';result.write_text(json.dumps(value));bundle.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'Review bundle changed'):load_manual_review(root,location,result)


if __name__=='__main__':unittest.main()
