import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from room_reconstruction import pipeline_layout as layout
from room_reconstruction.provenance import create_run_dir


class LayoutTests(unittest.TestCase):
    def test_new_jobs_and_legacy_discovery_without_moving_files(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            new=layout.create_job_dir(root,'new_job')
            self.assertEqual(new,root/'outputs/jobs/new_job')
            self.assertEqual(layout.job_dir(root,'new_job'),new)
            legacy=root/'outputs/old_job';legacy.mkdir()
            self.assertEqual(layout.job_dir(root,'old_job'),legacy)
            with self.assertRaises(FileExistsError):layout.create_job_dir(root,'old_job')
            with self.assertRaises(ValueError):layout.create_job_dir(root,'jobs')

    def test_reserved_nested_attempt_is_created_by_existing_provenance_producer(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();job=layout.create_job_dir(root,'capture')
            run=layout.reserve_run(root,job,'S03','capture_stage_A001')
            self.assertEqual(run,job/'stages/S03/capture_stage_A001')
            self.assertFalse(run.exists())
            self.assertEqual(create_run_dir(root,run.name),run)
            self.assertEqual(layout.resolve_run_dir(root,run.name),run)
            with self.assertRaises(FileExistsError):layout.reserve_run(root,job,'S03',run.name)
            with self.assertRaises(FileExistsError):create_run_dir(root,run.name)

    def test_index_escape_symlink_and_legacy_collision_rejected(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();job=layout.create_job_dir(root,'capture')
            index=root/'outputs/.run-index';index.mkdir()
            name='bad_A001';file=index/(name+'.json')
            file.write_text(json.dumps({'schema_version':1,'path':'../escape/'+name}))
            with self.assertRaises(ValueError):layout.resolve_run_dir(root,name)
            file.write_text(json.dumps({'schema_version':1,'path':'outputs/jobs/capture/stages/S03/'+name}))
            (job/'stages').symlink_to(Path(tmp).parent,target_is_directory=True)
            with self.assertRaises(ValueError):layout.resolve_run_dir(root,name)
            (job/'stages').unlink()
            (root/'outputs'/name).mkdir()
            with self.assertRaises(ValueError):layout.resolve_run_dir(root,name)

    def test_old_run_resolution_and_job_assets(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();legacy=root/'outputs/old';legacy.mkdir(parents=True)
            self.assertEqual(layout.resolve_run_dir(root,'old'),legacy)
            self.assertEqual(layout.job_asset(root,legacy,'logs','worker.log'),legacy/'worker.log')
            new=layout.create_job_dir(root,'new')
            self.assertEqual(layout.job_asset(root,new,'logs','worker.log'),new/'logs/worker.log')
            self.assertEqual(layout.job_asset(root,new,'config','machine.yaml'),new/'config/machine.yaml')


if __name__=='__main__':unittest.main()
