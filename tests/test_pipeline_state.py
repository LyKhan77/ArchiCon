import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from room_reconstruction.pipeline_state import StageRunner, FileLock, WaitingResources, StageExecutionError, fingerprint


CHILD = '''import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]);run=root/'outputs'/sys.argv[2];run.mkdir()
time.sleep(float(sys.argv[3]))
(run/'value.txt').write_text('actual child output')
(run/'status.json').write_text(json.dumps({'status':sys.argv[4]}))
'''


class StageRunnerTests(unittest.TestCase):
    def test_long_job_names_with_same_prefix_get_distinct_runs(self):
        from room_reconstruction.pipeline_layout import create_job_dir
        source=str(Path(__file__).resolve().parents[1]/'src')
        code="import sys,json;sys.path.insert(0,sys.argv[3]);from room_reconstruction.provenance import create_run_dir;run=create_run_dir(sys.argv[1],sys.argv[2]);(run/'status.json').write_text(json.dumps({'status':'complete'}))"
        command=[sys.executable,'-c',code,str(self.root),'@RUN_ID@',source]
        paths=[]
        for suffix in ['one','two']:
            directory=create_job_dir(self.root,'a'*24+suffix)
            path=StageRunner(self.root,directory,self.runner.config).run('S02',[self.input],{},command)
            self.assertTrue(path.is_relative_to(directory/'stages/S02'));paths.append(path)
        self.assertNotEqual(paths[0].name,paths[1].name)

    def test_nested_real_producer_resume_and_retry_are_job_scoped(self):
        from room_reconstruction.pipeline_layout import create_job_dir,resolve_run_dir
        job=create_job_dir(self.root,'nested_job');runner=StageRunner(self.root,job,self.runner.config)
        source=str(Path(__file__).resolve().parents[1]/'src')
        code="import sys,json;sys.path.insert(0,sys.argv[3]);from room_reconstruction.provenance import create_run_dir;run=create_run_dir(sys.argv[1],sys.argv[2]);(run/'value.txt').write_text('real');(run/'status.json').write_text(json.dumps({'status':'complete'}))"
        command=[sys.executable,'-c',code,str(self.root),'@RUN_ID@',source]
        first=runner.run('S02',[self.input],{},command,expected=('value.txt',))
        self.assertTrue(first.is_relative_to(job/'stages/S02'))
        self.assertEqual(resolve_run_dir(self.root,first.name),first)
        self.assertEqual(first,runner.run('S02',[self.input],{},command,expected=('value.txt',)))
        self.input.write_text('changed')
        second=runner.run('S02',[self.input],{},command,expected=('value.txt',))
        self.assertNotEqual(first,second)
        self.assertTrue((job/'logs'/(first.name+'.log')).is_file())

    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.job=self.root/'outputs/job';self.job.mkdir(parents=True)
        self.input=self.root/'input.txt';self.input.write_text('input')
        self.runner=StageRunner(self.root,self.job,{'compute':{'min_host_available_gib':.001,'min_disk_free_gib':.001}})

    def command(self,sleep='0',status='complete'):
        return [sys.executable,'-c',CHILD,str(self.root),'@RUN_ID@',sleep,status]

    def run_stage(self,**kwargs):
        return self.runner.run('S02',[self.input],{},self.command(),expected=('value.txt',),**kwargs)

    def test_actual_child_output_is_reused_only_while_unchanged(self):
        first=self.run_stage();self.assertEqual(first,self.run_stage())
        (first/'value.txt').write_text('tampered')
        second=self.run_stage();self.assertNotEqual(first,second)
        self.assertEqual((second/'value.txt').read_text(),'actual child output')

    def test_changed_input_invalidates_successful_stage(self):
        first=self.run_stage();self.input.write_text('new input')
        self.assertNotEqual(first,self.run_stage())

    def test_directory_input_hashes_members_and_invalidates_on_change(self):
        directory=self.root/'camera';directory.mkdir();member=directory/'cameras.json';member.write_text('{}')
        first=self.runner.run('S05',[directory],{},self.command(),expected=('value.txt',))
        self.assertEqual(first,self.runner.run('S05',[directory],{},self.command(),expected=('value.txt',)))
        member.write_text('{"pose":"changed"}')
        self.assertNotEqual(first,self.runner.run('S05',[directory],{},self.command(),expected=('value.txt',)))

    def test_changed_command_and_params_have_different_fingerprints(self):
        first=self.run_stage()
        second=self.runner.run('S02',[self.input],{'variant':2},self.command(),expected=('value.txt',))
        self.assertNotEqual(first,second)

    def test_failed_status_is_not_promoted_and_retry_gets_new_run(self):
        with self.assertRaises(RuntimeError):self.runner.run('S02',[self.input],{},self.command(status='failed'))
        path=self.run_stage();self.assertIn('A002',path.name)

    def test_timeout_terminates_owned_child_and_records_failure(self):
        with self.assertRaises(RuntimeError):self.runner.run('S02',[self.input],{},self.command(sleep='10'),timeout=.2)
        state=json.loads((self.job/'stages.json').read_text())
        self.assertEqual(state['S02'][-1]['status'],'failed')

    def test_nonblocking_lock_and_cleanup(self):
        path=self.job/'test.lock'
        with FileLock(path):
            with self.assertRaises(RuntimeError):
                with FileLock(path):pass
        with FileLock(path):pass

    def test_gpu_lock_contention_is_waiting_without_launching_child(self):
        uuid='GPU-11111111-1111-1111-1111-111111111111'
        runner=StageRunner(self.root,self.job,{'compute':{'gpu_selector':uuid}})
        with FileLock(self.root/'outputs'/'.locks'/f'{uuid}.lock'):
            with self.assertRaises(WaitingResources):runner.run('S05',[self.input],{},self.command(),gpu=True)

    def test_child_resource_failure_is_not_a_geometry_quality_failure(self):
        error=StageExecutionError('failed',self.root,{'error_type':'ValueError','reason':'Host MemAvailable below required 8 GiB'})
        self.assertTrue(error.resource_wait);self.assertFalse(error.quality_failure)

    def test_path_escape_and_nonfinite_config_rejected(self):
        with self.assertRaises(ValueError):self.runner.run('S02',[Path('/etc/passwd')],{},self.command())
        with self.assertRaises(ValueError):fingerprint({'parameter':float('nan')})


if __name__=='__main__':unittest.main()
