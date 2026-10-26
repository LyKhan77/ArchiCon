import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from room_reconstruction import pipeline_steps as steps
from room_reconstruction.provenance import sha256_file


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def component(root, name, ids):
    directory = root / 'outputs' / 'camera' / 'components' / name
    frames, native = [], []
    for i, fid in enumerate(ids):
        image = directory / 'undistorted/images' / (fid + '.jpg')
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(fid.encode())
        source = dict(frame_id=fid, source_index=i, timestamp_seconds=i * .5, split='train')
        pose = np.eye(4); pose[0, 3] = i
        native.append(source)
        frames.append(dict(source, audit_split='train', reconstruction_role='production',
            image_name=image.name, image_sha256=sha256_file(image), image_size=[20, 30],
            camera_model='PINHOLE', K=[[15, 0, 10], [0, 15, 15], [0, 0, 1]], c2w=pose.tolist()))
    write(directory / 'status.json', dict(stage='camera_full_component', status='camera_passed', camera_ready=True))
    write(directory / 'report.json', dict(stage='camera_full_component', quality=dict(passed=True), undistorted=True))
    write(directory / 'manifest.json', dict(stage='camera_full_component', selected_frames=native, source_sha256='video-sha'))
    write(directory / 'undistorted/cameras.json', dict(schema_version=1,
        pose_convention='OpenCV c2w; +x right, +y down, +z forward', image_root='images', frames=frames))
    return directory, frames


class PipelineStepsTests(unittest.TestCase):
    def test_global_stage_no_alignment_preserves_existing_tour_and_transfers_evidence(self):
        from room_reconstruction.global_architecture import build_architecture
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); job, _ = self.fixture(root)
            job['processing'] = dict(global_alignment=True)
            runner = unittest.mock.Mock()
            def execute(stage, inputs, params, command, **kwargs):
                self.assertEqual(stage, 'S06G_global')
                run = root / 'outputs/global'; run.mkdir(); products = run / 'products'; products.mkdir()
                report = build_architecture(root, products, _read(root / params['config']))
                write(run / 'status.json', dict(stage='global_architecture', status='complete'))
                write(run / 'report.json', report)
                return run
            runner.run.side_effect = execute
            bundle = steps.run_server(root, root / 'outputs/test_job', job, runner)
            self.assertEqual(bundle['global_architecture']['status'], 'no_alignment')
            self.assertIsNone(bundle['global_architecture']['component'])
            self.assertEqual(bundle['tour_run'], 'tour')
            self.assertTrue(any(item['file'].endswith('products/alignment.json') for item in bundle['artifacts']))
            self.assertEqual(bundle['coverage']['processed'], 3)

    def test_global_root_can_be_an_accepted_area_when_main_is_absent(self):
        from room_reconstruction.global_architecture import build_architecture
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); job, _ = self.fixture(root)
            job['processing'] = dict(global_alignment=True)
            job['reuse']['components'][0]['id'] = 'room'
            self.refresh_tour(root, job)
            runner = unittest.mock.Mock()
            def execute(stage, inputs, params, command, **kwargs):
                config = _read(root / params['config'])
                self.assertEqual(config['root_component'], 'room')
                run = root/'outputs/global'; run.mkdir(); products = run/'products'; products.mkdir()
                report = build_architecture(root, products, config)
                write(run/'status.json', dict(stage='global_architecture', status='complete'))
                write(run/'report.json', report)
                return run
            runner.run.side_effect = execute
            bundle = steps.run_server(root, root/'outputs/test_job', job, runner)
            self.assertEqual(bundle['global_architecture']['aligned_components'], ['room'])

    def test_bundle_transfers_registry_for_nested_export(self):
        from room_reconstruction.pipeline_layout import create_job_dir,reserve_run
        with TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();job,_=self.fixture(root)
            old=root/'outputs/tour';temporary=root/'saved-tour';old.rename(temporary)
            new_job=create_job_dir(root,'nested')
            target=reserve_run(root,new_job,'S07','tour');target.parent.mkdir(parents=True)
            temporary.rename(target)
            result=steps.run_server(root,root/'outputs/test_job',job,unittest.mock.Mock())
            self.assertIn('outputs/.run-index/tour.json',{a['file'] for a in result['artifacts']})

    def test_auto_camera_matching_uses_bounded_exhaustive_for_short_captures(self):
        self.assertEqual(steps.camera_matching_mode({},302),'exhaustive')
        self.assertEqual(steps.camera_matching_mode({},400),'exhaustive')
        self.assertEqual(steps.camera_matching_mode({},401),'sequential')
        self.assertEqual(steps.camera_matching_mode({'matching':'auto','max_exhaustive_frames':60},61),'sequential')

    def test_explicit_matching_override_and_invalid_auto_budget(self):
        self.assertEqual(steps.camera_matching_mode({'matching':'sequential'},24),'sequential')
        self.assertEqual(steps.camera_matching_mode({'matching':'exhaustive'},1000),'exhaustive')
        for config,count in [({'matching':'unknown'},60),({'max_exhaustive_frames':0},60),({'max_exhaustive_frames':True},60),({},0)]:
            with self.assertRaises(ValueError):steps.camera_matching_mode(config,count)

    def test_capture_cap_uses_whole_duration(self):
        capture = steps.capture_config('raw_videos/long.mov', 'sha', 700.25, 2)
        self.assertEqual(capture['verified_metadata']['sha256'], 'sha')
        self.assertGreaterEqual(capture['sampling']['max_keyframes'], 1401)
        self.assertEqual(capture['input_video'], 'raw_videos/long.mov')
        with self.assertRaises(ValueError): steps.capture_config('video', 'sha', float('nan'), 2)

    def test_capture_config_records_adaptive_sampling(self):
        capture = steps.capture_config('raw_videos/fast.mov', 'sha', 10, 2,
                                       sampling_mode='adaptive', adaptive_target_fps=4,
                                       motion_threshold=.08)
        self.assertEqual(capture['sampling']['mode'], 'adaptive')
        self.assertEqual(capture['sampling']['adaptive_target_fps'], 4)
        self.assertEqual(capture['sampling']['motion_threshold'], .08)

    def test_select_components_uses_quality_and_unique_frame_coverage(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, _ = component(root, 'a', ['f0', 'f1', 'f2', 'f3'])
            b, _ = component(root, 'b', ['f1', 'f2', 'f3'])
            c, _ = component(root, 'c', ['f4', 'f5', 'f6'])
            report = dict(components=[dict(path=str(p.relative_to(root / 'outputs/camera')), accepted=True)
                                      for p in (b, c, a)])
            selected, quarantined = steps.select_components(root, root / 'outputs/camera', report)
            self.assertEqual([row['camera_component'] for row in selected],
                             [str(p.relative_to(root)) for p in (a, c)])
            self.assertEqual(len(quarantined), 1)

    def test_select_components_keeps_partial_overlap_in_separate_worlds(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, _ = component(root, 'a', ['f0', 'f1', 'f2', 'f3'])
            b, _ = component(root, 'b', ['f3', 'f4', 'f5'])
            report = dict(components=[dict(path='components/a', accepted=True), dict(path='components/b', accepted=True)])
            selected, quarantined = steps.select_components(root, root / 'outputs/camera', report)
            self.assertEqual(len(selected), 2)
            self.assertEqual(quarantined, [])

    def test_new_annotations_bind_current_images_and_remain_unreviewed(self):
        with TemporaryDirectory() as tmp:
            directory, frames = component(Path(tmp), 'a', ['f0', 'f1', 'f2'])
            doc = steps.annotation_config(frames)
            self.assertFalse(doc['manual_review_complete'])
            self.assertEqual(set(doc['frames']), {'f0', 'f1', 'f2'})
            self.assertEqual(doc['frames']['f0']['image_sha256'], frames[0]['image_sha256'])
            self.assertEqual(doc['frames']['f0']['glass_polygons'], [])

    def fixture(self, root):
        video = root / 'raw_videos/test.mov'; video.parent.mkdir(); video.write_bytes(b'video')
        digest = sha256_file(video)
        audit = root / 'outputs/audit'
        native = [dict(frame_id=f'f{i}', image_path=f'keyframes/f{i}.jpg') for i in range(5)]
        write(audit / 'status.json', dict(stage='video_audit', status='complete'))
        write(audit / 'frame_manifest.json', dict(source_sha256=digest, frames=native))
        write(audit / 'audit_report.json', dict(source_sha256=digest, contact_sheets=[]))
        camera, frames = component(root, 'a', ['f0', 'f1', 'f2'])
        manifest = json.loads((camera / 'manifest.json').read_text()); manifest['source_sha256'] = digest
        write(camera / 'manifest.json', manifest)
        write(root / 'outputs/camera/status.json', dict(stage='camera_full', status='camera_passed', camera_ready=True))
        write(root / 'outputs/camera/manifest.json', dict(source_sha256=digest, audit_run='audit',
            audit_manifest_sha256=sha256_file(audit / 'frame_manifest.json')))
        write(root / 'outputs/camera/report.json', dict(source_sha256=digest, components=[]))
        depth = root / 'outputs/depth'
        artifacts = []
        for frame in frames:
            path = depth / (frame['frame_id'] + '.npz'); path.parent.mkdir(exist_ok=True); path.write_bytes(b'depth')
            artifacts.append(dict(frame_id=frame['frame_id'], file=path.name, sha256=sha256_file(path),
                                  timestamp_seconds=frame['timestamp_seconds']))
        relative = str(camera.relative_to(root))
        write(depth / 'status.json', dict(stage='full_depth', status='depth_passed', depth_ready=True))
        write(depth / 'manifest.json', dict(stage='full_depth', camera_component=relative, selected_frames=frames,
            upstream_hashes={str(camera / 'undistorted/cameras.json'): sha256_file(camera / 'undistorted/cameras.json')}))
        write(depth / 'report.json', dict(stage='full_depth', status='depth_passed', camera_component=relative,
            unique_frame_count=3, artifacts=artifacts))
        surface = root / 'outputs/surface'
        write(surface / 'status.json', dict(stage='full_surface', status='mesh_passed', mesh_ready=True))
        write(surface / 'manifest.json', dict(stage='full_surface', input_and_source_hashes={
            str(depth / 'report.json'): sha256_file(depth / 'report.json')}))
        write(surface / 'report.json', dict(stage='full_surface', status='mesh_passed',
            depth_run='depth', camera_component=relative, meshes={}))
        render = root / 'outputs/render'; render.mkdir()
        mesh = render / 'room-full.glb'; mesh.write_bytes(b'mesh')
        write(render / 'status.json', dict(stage='full_render', status='render_passed'))
        write(render / 'report.json', dict(stage='full_render', frames=[]))
        write(render / 'manifest.json', dict(stage='full_render', surface_run='surface', camera_component=relative,
            world_to_viewer=np.eye(4).tolist(), input_and_source_hashes={
                str(camera / 'undistorted/cameras.json'): sha256_file(camera / 'undistorted/cameras.json')},
            artifacts=[dict(file=mesh.name, sha256=sha256_file(mesh), size_bytes=mesh.stat().st_size)]))
        tour = root / 'outputs/tour'
        data = write(tour / 'tour-data.json', dict(components=[dict(id='main', stats=dict(viewpoints=3))]))
        write(tour / 'status.json', dict(stage='room_tour_export', status='exported'))
        write(tour / 'manifest.json', dict(stage='room_tour_export', input_hashes={
            str(mesh.relative_to(root)): sha256_file(mesh),
            str((render / 'manifest.json').relative_to(root)): sha256_file(render / 'manifest.json'),
            str((camera / 'undistorted/cameras.json').relative_to(root)): sha256_file(camera / 'undistorted/cameras.json')},
            artifacts=[dict(file=data.name, sha256=sha256_file(data), size_bytes=data.stat().st_size)]))
        reuse = dict(audit_run='audit', camera_run='camera', tour_run='tour', components=[dict(id='main', name='Main',
            camera_component=relative, depth_run='depth', surface_run='surface', render_run='render')])
        return dict(job_id='test_job', input_video='raw_videos/test.mov', input_sha256=digest,
                    machine_config='configs/server.yaml', reuse=reuse), mesh

    def add_world(self, root, job, name, ids):
        camera, frames = component(root, name, ids)
        meta = _read(camera / 'manifest.json'); meta['source_sha256'] = job['input_sha256']; write(camera / 'manifest.json', meta)
        row = dict(id=name, name=name, camera_component=str(camera.relative_to(root)),
                   depth_run='depth_' + name, surface_run='surface_' + name, render_run='render_' + name)
        depth=root / 'outputs' / row['depth_run']; artifacts=[]
        for frame in frames:
            path=depth / (frame['frame_id']+'.npz'); path.parent.mkdir(exist_ok=True); path.write_bytes(b'depth')
            artifacts.append(dict(frame_id=frame['frame_id'], file=path.name, sha256=sha256_file(path)))
        write(depth / 'status.json',dict(stage='full_depth',status='depth_passed',depth_ready=True))
        write(depth / 'manifest.json',dict(camera_component=row['camera_component'],selected_frames=frames,
            upstream_hashes={str(camera / 'undistorted/cameras.json'):sha256_file(camera / 'undistorted/cameras.json')}))
        write(depth / 'report.json',dict(status='depth_passed',camera_component=row['camera_component'],
            unique_frame_count=len(frames),artifacts=artifacts))
        surface=root / 'outputs' / row['surface_run']
        write(surface / 'status.json',dict(stage='full_surface',status='mesh_passed'))
        write(surface / 'manifest.json',dict(input_and_source_hashes={}))
        write(surface / 'report.json',dict(depth_run=row['depth_run'],camera_component=row['camera_component'],meshes={}))
        render=root / 'outputs' / row['render_run']; render.mkdir(); mesh=render / 'room-full.glb'; mesh.write_bytes(b'mesh'+name.encode())
        write(render / 'status.json',dict(stage='full_render',status='render_passed'))
        write(render / 'report.json',dict(frames=[]))
        write(render / 'manifest.json',dict(surface_run=row['surface_run'],camera_component=row['camera_component'],
            input_and_source_hashes={str(camera / 'undistorted/cameras.json'):sha256_file(camera / 'undistorted/cameras.json')},
            artifacts=[dict(file=mesh.name,sha256=sha256_file(mesh))]))
        job['reuse']['components'].append(row)
        self.refresh_tour(root,job)
        return row

    def refresh_tour(self, root, job):
        hashes={}; components=[]
        for row in job['reuse']['components']:
            render=root / 'outputs' / row['render_run']
            for path in (render / 'room-full.glb',render / 'manifest.json',root / row['camera_component'] / 'undistorted/cameras.json'):
                hashes[str(path.relative_to(root))]=sha256_file(path)
            components.append(dict(id=row['id'],stats=dict(viewpoints=3)))
        tour=root / 'outputs/tour'; data=write(tour / 'tour-data.json',dict(components=components))
        write(tour / 'manifest.json',dict(input_hashes=hashes,artifacts=[dict(file=data.name,sha256=sha256_file(data))]))

    def evidence(self,root,row,size=16):
        render=root / 'outputs' / row['render_run']; records=[]
        for name in ['overview.png','full_path_review.jpg']:
            (render / name).write_bytes(b'x'*size)
        for i in range(10):
            path=render / f'frames/f{i}.png';path.parent.mkdir(exist_ok=True);path.write_bytes(b'x'*size)
            records.append(dict(file=str(path.relative_to(render)),sha256=sha256_file(path)))
        write(render / 'report.json',dict(frames=records))

    def test_reuse_partial_overlap_counts_each_audited_id_once(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();job,_=self.fixture(root)
            self.add_world(root,job,'second',['f2','f3','f4'])
            result=steps.run_server(root,root / 'outputs/test_job',job,unittest.mock.Mock())
            self.assertEqual(len(result['components']),2)
            self.assertEqual(result['coverage']['processed'],5)
            self.assertEqual(result['coverage']['gaps'],[])

    def test_bad_component_preserves_valid_world_and_reports_its_unique_gaps(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();job,_=self.fixture(root)
            bad=self.add_world(root,job,'bad',['f2','f3','f4'])
            path=root / 'outputs/depth_bad/manifest.json';doc=_read(path);doc['selected_frames']=[];write(path,doc)
            def export(*args,**kwargs):
                accepted=dict(job,reuse=dict(job['reuse'],components=job['reuse']['components'][:1]))
                self.refresh_tour(root,accepted)
                return root / 'outputs/tour'
            runner=unittest.mock.Mock();runner.run.side_effect=export
            result=steps.run_server(root,root / 'outputs/test_job',job,runner)
            self.assertEqual([c['id'] for c in result['components']],['main'])
            self.assertEqual(result['coverage']['gaps'],['f3','f4'])
            self.assertEqual(result['failed_components'][0]['id'],'bad')
            self.assertIn('provenance',result['failed_components'][0]['reason'])

    def test_all_failed_components_write_diagnostics_without_export(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();job,mesh=self.fixture(root);mesh.write_bytes(b'tampered')
            runner=unittest.mock.Mock()
            with self.assertRaises(ValueError):steps.run_server(root,root / 'outputs/test_job',job,runner)
            runner.run.assert_not_called()
            diagnostic=_read(root / 'outputs/test_job/component-failures.json')
            self.assertEqual(diagnostic['failed_components'][0]['id'],'main')
            self.assertEqual(diagnostic['coverage']['processed'],0)

    def test_component_infrastructure_failures_are_not_silently_quarantined(self):
        from room_reconstruction.pipeline_state import WaitingResources
        for fault in (FileNotFoundError('missing runtime'),WaitingResources('GPU occupied'),RuntimeError('nvidia-smi failed')):
            with self.subTest(fault=fault),TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();job,_=self.fixture(root)
                del job['reuse']['components'][0]['depth_run']
                runner=unittest.mock.Mock();runner.run.side_effect=fault
                with self.assertRaises(type(fault)):
                    steps.run_server(root,root / 'outputs/test_job',job,runner)

    def test_typed_child_data_failure_preserves_its_diagnostic_reason(self):
        from room_reconstruction.pipeline_state import StageExecutionError
        error=StageExecutionError('Stage exited 2',Path('/unused'),
                                  dict(error_type='ValueError',reason='No supported component surface'))
        runner=unittest.mock.Mock();runner.run.side_effect=error
        with self.assertRaisesRegex(steps.ComponentDataError,'supported component surface'):
            steps._component_stage(runner,'S06',[],{},[])

    def test_corrupt_render_panel_isolated_before_bundle_assembly(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();job,_=self.fixture(root)
            bad=self.add_world(root,job,'bad',['f2','f3','f4']);self.evidence(root,bad)
            (root / 'outputs/render_bad/frames/f0.png').write_bytes(b'changed')
            def export(*args,**kwargs):
                accepted=dict(job,reuse=dict(job['reuse'],components=job['reuse']['components'][:1]))
                self.refresh_tour(root,accepted);return root / 'outputs/tour'
            runner=unittest.mock.Mock();runner.run.side_effect=export
            result=steps.run_server(root,root / 'outputs/test_job',job,runner)
            self.assertEqual([c['id'] for c in result['components']],['main'])
            self.assertIn('SHA',result['failed_components'][0]['reason'])

    def test_qa_reserves_evidence_for_every_component_before_extra_images(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();job,_=self.fixture(root)
            self.add_world(root,job,'second',['f2','f3','f4'])
            self.add_world(root,job,'third',['f0','f3','f4'])
            for row in job['reuse']['components']:self.evidence(root,row)
            result=steps.run_server(root,root / 'outputs/test_job',job,unittest.mock.Mock())
            self.assertLessEqual(len(result['qa']),32)
            for row in job['reuse']['components']:
                evidence=result['qa_component_evidence'][row['id']]
                self.assertTrue(evidence['complete'])
                self.assertTrue(evidence['source'])
                self.assertTrue(evidence['render'])
                self.assertTrue(evidence['overview'])
                self.assertTrue(all(p in result['qa'] for p in evidence['source']+evidence['render']+evidence['overview']))

    def test_qa_budget_cannot_silently_omit_required_component_evidence(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();job,_=self.fixture(root)
            self.evidence(root,job['reuse']['components'][0],size=21*1024**2)
            result=steps.run_server(root,root / 'outputs/test_job',job,unittest.mock.Mock())
            evidence=result['qa_component_evidence']['main']
            self.assertFalse(evidence['complete'])
            self.assertIn('budget',evidence['reason'])

    def test_reuse_exports_union_coverage_and_verified_workspace_artifacts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); job, mesh = self.fixture(root)
            runner = unittest.mock.Mock()
            result = steps.run_server(root, root / 'outputs/test_job', job, runner)
            runner.run.assert_not_called()
            self.assertEqual(result['coverage'], dict(attempted=5, processed=3, gaps=['f3', 'f4'],
                                                      full_room_reconstruction_complete=False))
            self.assertEqual(result['components'][0]['mesh'], str(mesh.relative_to(root)))
            self.assertTrue(all(not Path(a['file']).is_absolute() for a in result['artifacts']))
            self.assertTrue((root / 'outputs/test_job/bundle.json').is_file())

    def test_reuse_rejects_changed_render_mesh(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); job, mesh = self.fixture(root); mesh.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, '[Hh]ash|SHA'):
                steps.run_server(root, root / 'outputs/test_job', job, unittest.mock.Mock())

    def test_reused_tour_mesh_may_be_identical_delivery_copy(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); job, mesh = self.fixture(root)
            delivery = root / 'outputs/delivery/copy.glb'; delivery.parent.mkdir(); delivery.write_bytes(mesh.read_bytes())
            path = root / 'outputs/tour/manifest.json'; doc = json.loads(path.read_text())
            del doc['input_hashes'][str(mesh.relative_to(root))]
            doc['input_hashes'][str(delivery.relative_to(root))] = sha256_file(delivery); write(path, doc)
            result = steps.run_server(root, root / 'outputs/test_job', job, unittest.mock.Mock())
            self.assertEqual(result['components'][0]['mesh'], str(mesh.relative_to(root)))

    def export_module(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/export_room_tour.py'
        spec = importlib.util.spec_from_file_location('pipeline_tour_export', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module

    def test_tour_limitations_use_current_coverage_and_names(self):
        exporter = self.export_module()
        result = exporter.export_limitations(dict(coverage=dict(attempted=8, processed=6, gaps=['f6', 'f7'])),
            [dict(name='Laboratorium', stats=dict(viewpoints=6))])
        self.assertIn('6 dari 8', ' '.join(result))
        self.assertNotIn('283', ' '.join(result))
        self.assertNotIn('Pintu masuk', ' '.join(result))
        self.assertEqual(exporter.viewer_divisor(dict(viewer_voxel_divisor=650)), 650)
        with self.assertRaises(ValueError): exporter.viewer_divisor(dict(viewer_voxel_divisor=0))

    def test_tour_floor_fallback_is_explicit_when_no_horizontal_support(self):
        exporter = self.export_module()
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [1, 2, 0]], float)
        floor, policy, supported = exporter.floor_estimate(vertices, np.array([[0, 1, 2], [1, 3, 2]]), [1., 1.1])
        self.assertLess(floor, 1.)
        self.assertFalse(supported)
        self.assertIn('unsupported', policy)

    def test_reuse_rejects_wrong_video_and_camera_provenance(self):
        for fault in ('video', 'camera'):
            with self.subTest(fault=fault), TemporaryDirectory() as tmp:
                root = Path(tmp); job, _ = self.fixture(root)
                if fault == 'video': job['input_sha256'] = 'bad'
                else:
                    path = root / 'outputs/render/manifest.json'; doc = json.loads(path.read_text())
                    doc['input_and_source_hashes'] = {}; write(path, doc)
                with self.assertRaises(ValueError):
                    steps.run_server(root, root / 'outputs/test_job', job, unittest.mock.Mock())

    def test_depth_reuse_requires_recorded_current_camera_hash(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); job, _ = self.fixture(root)
            path = root / 'outputs/depth/manifest.json'; doc = _read(path); doc['upstream_hashes'] = {}; write(path, doc)
            with self.assertRaisesRegex(ValueError, 'camera'):
                steps.run_server(root, root / 'outputs/test_job', job, unittest.mock.Mock())

    def test_qa_samples_contact_sheets_to_leave_space_for_source_frames(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); job, _ = self.fixture(root)
            audit = root / 'outputs/audit'; names=[]
            for i in range(40):
                path=audit / f'review/sheet_{i}.jpg'; path.parent.mkdir(exist_ok=True); path.write_bytes(b'image')
                names.append(str(path.relative_to(audit)))
            path=audit / 'audit_report.json'; doc=_read(path); doc['contact_sheets']=names; write(path,doc)
            result = steps.run_server(root, root / 'outputs/test_job', job, unittest.mock.Mock())
            self.assertTrue(any('/undistorted/images/' in p for p in result['qa']))
            self.assertLessEqual(sum('/review/' in p for p in result['qa']), 6)

    def test_stage_command_uses_isolated_runtimes_and_config(self):
        root = Path('/work')
        command = steps.stage_command(root, 'depth', camera_component='outputs/camera', config='configs/custom.yaml')
        self.assertEqual(command[0], '/work/.venv-inference/bin/python')
        self.assertIn('--config', command)
        self.assertEqual(command[command.index('--config') + 1], 'configs/custom.yaml')
        self.assertIn('@RUN_ID@', command)
        self.assertEqual(steps.stage_command(root, 'surface', depth_run='d', camera_component='outputs/c',
                         annotations='outputs/j/a.json')[0], '/work/.venv-geometry/bin/python')

    def test_depth_cli_accepts_machine_config(self):
        import sys
        path = Path(__file__).resolve().parents[1] / 'scripts/run_full_depth.py'
        spec = importlib.util.spec_from_file_location('pipeline_depth', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with patch.object(sys, 'argv', ['depth', '--camera-component', 'outputs/c', '--run-id', 'new',
                                      '--config', 'outputs/j/machine.server.yaml']), \
             patch.object(module, 'create_run_dir', return_value=Path('/unused')), \
             patch.object(module, 'write_json'), \
             patch.object(module, 'execute', return_value=dict(status='depth_passed', unique_frame_count=3)) as execute:
            self.assertEqual(module.main(), 0)
        self.assertEqual(execute.call_args.args[1].config, 'outputs/j/machine.server.yaml')

    def test_depth_and_render_cli_failures_preserve_typed_data_diagnostics(self):
        import sys
        from room_reconstruction.pipeline_state import StageExecutionError
        scripts=Path(__file__).resolve().parents[1] / 'scripts'
        for script,arguments in [('run_full_depth.py',['--camera-component','outputs/c']),
                                 ('render_full_surface.py',['--surface-run','surface'])]:
            with self.subTest(script=script),TemporaryDirectory() as tmp:
                spec=importlib.util.spec_from_file_location('typed_'+script[:-3],scripts / script)
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                owner=module.fusion if script.startswith('render') else module
                run=Path(tmp)
                with patch.object(sys,'argv',[script,*arguments,'--run-id','typed']), \
                     patch.object(sys,'stderr',io.StringIO()), \
                     patch.object(owner,'create_run_dir',return_value=run), \
                     patch.object(module,'execute',side_effect=ValueError('Invalid component camera contract')):
                    self.assertEqual(module.main(),2)
                status=_read(run / 'status.json')
                self.assertEqual(status['status'],'failed')
                self.assertEqual(status['error_type'],'ValueError')
                self.assertEqual(status['reason'],'Invalid component camera contract')
                self.assertTrue(StageExecutionError('Stage exited 2',run,status).quality_failure)

    def test_fresh_path_plans_all_stages_and_preserves_long_video(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); job, _ = self.fixture(root); job['reuse'] = {}
            write(root / 'outputs/camera/report.json', dict(components=[dict(path='components/a', accepted=True)]))
            calls = []
            def execute(stage, inputs, params, command, **options):
                calls.append((stage, inputs, params, command, options))
                return root / 'outputs' / {'S02_audit': 'audit', 'S03_camera': 'camera',
                    'S05_depth_main': 'depth', 'S06_surface_main': 'surface',
                    'S06R_render_main': 'render', 'S07_tour': 'tour'}[stage]
            runner = unittest.mock.Mock(); runner.run.side_effect = execute
            probe = unittest.mock.Mock(stdout=json.dumps(dict(format=dict(duration='900.25'))))
            with patch.object(steps.subprocess, 'run', return_value=probe):
                result = steps.run_server(root, root / 'outputs/test_job', job, runner)
            self.assertEqual(len(calls), 6)
            self.assertEqual([c[0] for c in calls], ['S02_audit', 'S03_camera', 'S05_depth_main',
                                                    'S06_surface_main', 'S06R_render_main', 'S07_tour'])
            self.assertEqual([c[0] for c in calls if c[4].get('gpu')], ['S05_depth_main'])
            self.assertEqual(calls[1][2]['matching'],'exhaustive')
            camera_command=calls[1][3]
            self.assertEqual(camera_command[camera_command.index('--matching')+1],'exhaustive')
            self.assertGreaterEqual(_read(root / 'outputs/test_job/capture.json')['sampling']['max_keyframes'], 1801)
            self.assertNotIn('--max-windows', calls[2][3])
            self.assertEqual(_read(root / 'outputs/test_job/tour-config.json')['coverage'], result['coverage'])

    def test_refinement_invalidates_only_requested_downstream_stages(self):
        for start, expected in [('S06', ['S06_surface_main', 'S06R_render_main', 'S07_tour']), ('S07', ['S07_tour'])]:
            with self.subTest(start=start), TemporaryDirectory() as tmp:
                root = Path(tmp).resolve(); job, _ = self.fixture(root)
                job.update(refinement_start=start, processing=dict(confidence_quantile=.05, viewer_voxel_divisor=650))
                calls = []
                def execute(stage, inputs, params, command, **options):
                    calls.append((stage, params))
                    return root / 'outputs' / {'S06_surface_main': 'surface', 'S06R_render_main': 'render', 'S07_tour': 'tour'}[stage]
                runner = unittest.mock.Mock(); runner.run.side_effect = execute
                steps.run_server(root, root / 'outputs/test_job', job, runner)
                self.assertEqual([c[0] for c in calls], expected)
                self.assertEqual(_read(root / 'outputs/test_job/tour-config.json')['viewer_voxel_divisor'], 650)
                if start == 'S06': self.assertEqual(calls[0][1]['confidence_quantile'], .05)

    def test_refinement_keeps_upstream_camera_failures_in_bundle(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();job,_=self.fixture(root)
            prior=dict(id='rejected_camera',stage='S03_camera',camera_component='outputs/camera/rejected',
                       reason='Insufficient registered observations',error_type='ComponentDataError')
            job['reuse']['failed_components']=[prior]
            runner=unittest.mock.Mock();runner.run.return_value=root/'outputs/tour'
            result=steps.run_server(root,root/'outputs/test_job',job,runner)
            self.assertEqual(result.get('failed_components'),[prior])


def _read(path):
    return json.loads(path.read_text())


class PipelineWorkerTests(unittest.TestCase):
    def worker(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/pipeline_worker.py'
        spec = importlib.util.spec_from_file_location('pipeline_worker_test', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module

    def fixture(self, root):
        config = write(root / 'configs/server.json', dict(profile='server', paths=dict(project_root=str(root))))
        directory = root / 'outputs/job1'
        write(directory / 'job.json', dict(job_id='job1', machine_config=str(config.relative_to(root))))
        return directory

    def test_worker_persists_success_bundle_and_pid(self):
        worker = self.worker()
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); directory = self.fixture(root)
            with patch.object(worker, 'run_server', return_value=dict(schema_version=1)), patch.object(worker, 'StageRunner'):
                self.assertEqual(worker.process_job(root, 'job1'), 0)
            state = _read(directory / 'worker-status.json')
            self.assertEqual(state['execution_status'], 'complete')
            self.assertEqual(state['bundle'], 'outputs/job1/bundle.json')
            self.assertGreater(state['worker_pid'], 0)
            self.assertEqual(_read(directory / 'bundle.json'), dict(schema_version=1))

    def test_worker_records_exception_and_traceback(self):
        worker = self.worker()
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); directory = self.fixture(root)
            with patch.object(worker, 'run_server', side_effect=ValueError('bad mesh hash')), patch.object(worker, 'StageRunner'):
                self.assertEqual(worker.process_job(root, 'job1'), 1)
            self.assertEqual(_read(directory / 'worker-status.json')['execution_status'], 'failed')
            self.assertIn('bad mesh hash', (directory / 'worker.log').read_text())

    def test_worker_waiting_resources_is_resumable(self):
        worker = self.worker()
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); directory = self.fixture(root)
            with patch.object(worker, 'run_server', side_effect=worker.WaitingResources('GPU occupied')), patch.object(worker, 'StageRunner'):
                self.assertEqual(worker.process_job(root, 'job1'), 75)
            self.assertEqual(_read(directory / 'worker-status.json')['execution_status'], 'waiting_resources')

    def test_worker_rejects_invalid_id_and_preserves_active_worker_status(self):
        worker = self.worker()
        from room_reconstruction.pipeline_state import FileLock
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); directory = self.fixture(root)
            with self.assertRaises(ValueError): worker.process_job(root, '../escape')
            write(directory / 'worker-status.json', dict(execution_status='running', worker_pid=123))
            with FileLock(directory / 'worker.lock'):
                self.assertEqual(worker.process_job(root, 'job1'), 73)
            self.assertEqual(_read(directory / 'worker-status.json')['worker_pid'], 123)


if __name__ == '__main__': unittest.main()
