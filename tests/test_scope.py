import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import yaml

from room_reconstruction.pipeline_cli import register_job
from room_reconstruction.pipeline_steps import capture_config
from room_reconstruction.scope import normalize_scope


class ScopeContractTests(unittest.TestCase):
    def test_missing_scope_defaults_to_full_traversal(self):
        self.assertEqual(normalize_scope(None), {
            'kind': 'full_traversal',
            'name': None,
            'start_time_seconds': None,
            'end_time_seconds': None,
            'source_policy': 'full_input_video',
        })

    def test_single_room_scope_preserves_name_and_range(self):
        value = normalize_scope({
            'kind': 'single_room', 'name': ' Workshop ',
            'start_time_seconds': 2, 'end_time_seconds': 61,
        }, duration_seconds=68.99)
        self.assertEqual(value['name'], 'Workshop')
        self.assertEqual(value['start_time_seconds'], 2.0)
        self.assertEqual(value['end_time_seconds'], 61.0)

    def test_scope_rejects_invalid_kind_name_and_range(self):
        cases = [
            {'kind': 'unknown'},
            {'kind': 'single_room', 'name': '   '},
            {'kind': 'single_room', 'start_time_seconds': -1},
            {'kind': 'single_room', 'start_time_seconds': 8, 'end_time_seconds': 8},
            {'kind': 'single_room', 'start_time_seconds': 1, 'end_time_seconds': 70},
        ]
        for value in cases:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_scope(value, duration_seconds=68.99)

    def test_registration_persists_normalized_scope(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'configs').mkdir()
            (root / 'raw_videos').mkdir()
            video = root / 'raw_videos/workshop.MOV'
            video.write_bytes(b'workshop')
            machine = root / 'configs/machine.yaml'
            machine.write_text(yaml.safe_dump({
                'profile': 'server', 'paths': {'project_root': '/server/project'}, 'compute': {},
            }))
            profile = root / 'configs/pipeline.yaml'
            profile.write_text(yaml.safe_dump({
                'processing': {'target_fps': 2}, 'review': {'required': True},
            }))
            job = register_job(root, video, 'workshop_001', machine, profile,
                               ssh_target='user@server', scope={
                                   'kind': 'single_room', 'name': 'Workshop',
                               })
            saved = json.loads((root / 'outputs/jobs/workshop_001/job.json').read_text())
            self.assertEqual(saved['scope'], job['scope'])
            self.assertEqual(saved['scope']['kind'], 'single_room')
            self.assertEqual(saved['scope']['name'], 'Workshop')

    def test_capture_config_carries_scope_and_checks_video_duration(self):
        config = capture_config('raw_videos/workshop.MOV', 'a' * 64, 68.99,
                                scope={'kind': 'single_room', 'name': 'Workshop'})
        self.assertEqual(config['scope']['kind'], 'single_room')
        self.assertEqual(config['scope']['source_policy'], 'full_input_video')
        with self.assertRaises(ValueError):
            capture_config('raw_videos/workshop.MOV', 'a' * 64, 68.99,
                           scope={'kind': 'single_room', 'name': 'Workshop',
                                  'end_time_seconds': 69})

    def test_pipeline_profile_scope_is_used_when_cli_scope_is_omitted(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'configs').mkdir(); (root / 'raw_videos').mkdir()
            video = root / 'raw_videos/workshop.MOV'; video.write_bytes(b'workshop')
            machine = root / 'configs/machine.yaml'
            machine.write_text(yaml.safe_dump({'profile':'server','paths':{'project_root':'/server'},'compute':{}}))
            profile = root / 'configs/pipeline.yaml'
            profile.write_text(yaml.safe_dump({'scope':{'kind':'single_room','name':'Workshop'},
                                               'processing':{},'review':{'required':True}}))
            job = register_job(root, video, 'profile_scope', machine, profile, ssh_target='user@server')
            self.assertEqual(job['scope']['kind'], 'single_room')


if __name__ == '__main__':
    unittest.main()
