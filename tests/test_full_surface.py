import json
import importlib.util
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from room_reconstruction import full_surface
from room_reconstruction.depth_da3 import expected_resize
from room_reconstruction.provenance import sha256_file


class FullSurfaceTests(unittest.TestCase):
    def test_coverage_keeps_unknown_but_rejects_contradictions(self):
        base = np.array([[True, True, True, True, False]])
        support = np.array([[0, 0, 1, 2, 3]], np.uint16)
        contradictions = np.array([[0, 1, 1, 3, 0]], np.uint16)
        result = full_surface.coverage_masks(base, support, contradictions)
        np.testing.assert_array_equal(result['coverage_mask'], [[True, False, True, False, False]])
        np.testing.assert_array_equal(result['strict_mask'], [[False, False, True, False, False]])
        np.testing.assert_array_equal(result['uncertain_mask'], [[True, False, False, False, False]])

    def fixture(self, root, n=3):
        component = root/'outputs/camera/components/component_000'
        images = component/'undistorted/images'; images.mkdir(parents=True)
        native=[]; frames=[]
        for i in range(n):
            path=images/f'{i}.jpg';path.write_bytes(f'image{i}'.encode())
            pose=np.eye(4);pose[0,3]=i*.1
            source=dict(frame_id=f'f{i}',source_index=i,timestamp_seconds=i*.5,split='validation' if i%2 else 'train')
            native.append(source)
            frames.append(dict(source,split='train',audit_split=source['split'],reconstruction_role='production',
                image_name=path.name,image_sha256=sha256_file(path),image_size=[280,504],
                camera_model='PINHOLE',K=[[250.,0,140],[0,250,252],[0,0,1]],c2w=pose.tolist()))
        for name,data in dict(status=dict(stage='camera_full_component',status='camera_passed',camera_ready=True),
                manifest=dict(stage='camera_full_component',selected_frames=native),
                report=dict(stage='camera_full_component',quality=dict(passed=True),undistorted=True)).items():
            (component/f'{name}.json').write_text(json.dumps(data))
        (component/'undistorted/cameras.json').write_text(json.dumps(dict(schema_version=1,
            pose_convention='OpenCV c2w; +x right, +y down, +z forward',image_root='images',frames=frames)))
        depth=root/'outputs/depth';depth.mkdir()
        K,hw=expected_resize(np.array([f['K'] for f in frames]),frames[0]['image_size']);artifacts=[]
        for i,frame in enumerate(frames):
            path=depth/f'{frame["frame_id"]}.npz'
            np.savez_compressed(path,depth=np.full(hw,3.,np.float32),conf=np.ones(hw,np.float32),
                K=K[i],w2c=np.linalg.inv(frame['c2w'])[:3].astype(np.float32),
                input_c2w=np.array(frame['c2w'],np.float32),processed_rgb=np.zeros((*hw,3),np.uint8))
            artifacts.append(dict(frame_id=frame['frame_id'],timestamp_seconds=frame['timestamp_seconds'],
                file=path.name,sha256=sha256_file(path)))
        relative=str(component.relative_to(root))
        for name,data in dict(status=dict(stage='full_depth',status='depth_passed',depth_ready=True),
                manifest=dict(stage='full_depth',camera_component=relative,selected_frames=frames,upstream_hashes={}),
                report=dict(stage='full_depth',status='depth_passed',camera_component=relative,unique_frame_count=n,
                    units=full_surface.UNITS,artifacts=artifacts)).items():
            (depth/f'{name}.json').write_text(json.dumps(data))
        annotations=root/'annotations.json';annotations.write_text(json.dumps(dict(frames={f['frame_id']:
            dict(image_sha256=f['image_sha256'],manual_people_boxes=[],glass_polygons=[]) for f in frames})))
        return relative,annotations

    def test_variable_n_production_component_and_streamed_camera_validation(self):
        for n in (3,5):
            with self.subTest(n=n),TemporaryDirectory() as tmp:
                root=Path(tmp);component,annotations=self.fixture(root,n)
                inputs=full_surface.prepare_inputs(root,'depth',component,annotations)
                self.assertEqual(len(inputs['frames']),n)
                frame=full_surface.load_depth(inputs,0)
                self.assertEqual(frame['depth'].shape,(504,280))
                np.testing.assert_allclose(frame['w2c'],np.eye(4)[:3])

    def test_rejects_proper_but_wrong_camera_or_rgb_grid(self):
        for fault in ('K','w2c','input_c2w','processed_rgb'):
            with self.subTest(fault=fault),TemporaryDirectory() as tmp:
                root=Path(tmp);component,annotations=self.fixture(root)
                inputs=full_surface.prepare_inputs(root,'depth',component,annotations)
                path=inputs['paths'][0]
                with np.load(path,allow_pickle=False) as archive:values=dict(archive)
                if fault=='K':values[fault][0,0]*=1.1
                elif fault in ('w2c','input_c2w'):values[fault][0,3]+=1
                else:values[fault]=values[fault][:-1]
                np.savez_compressed(path,**values)
                # Keep the byte identity valid so this reaches the numeric camera/grid checks.
                inputs['artifacts'][0]['sha256']=sha256_file(path)
                message='RGB grid' if fault=='processed_rgb' else 'camera'
                with self.assertRaisesRegex(ValueError,message):full_surface.load_depth(inputs,0)

    def test_annotation_exact_coverage_and_image_hash_required(self):
        for fault in ('missing','extra','hash'):
            with self.subTest(fault=fault),TemporaryDirectory() as tmp:
                root=Path(tmp);component,annotations=self.fixture(root)
                value=json.loads(annotations.read_text())
                if fault=='missing':del value['frames']['f0']
                elif fault=='extra':value['frames']['extra']=value['frames']['f0']
                else:value['frames']['f0']['image_sha256']='0'*64
                annotations.write_text(json.dumps(value))
                with self.assertRaises(ValueError):full_surface.prepare_inputs(root,'depth',component,annotations)

    def test_rejects_partial_duplicate_and_mismatched_component_depth(self):
        for fault in ('partial','duplicate','component','selected'):
            with self.subTest(fault=fault),TemporaryDirectory() as tmp:
                root=Path(tmp);component,annotations=self.fixture(root)
                name='status' if fault=='partial' else 'manifest' if fault=='selected' else 'report'
                path=root/f'outputs/depth/{name}.json';value=json.loads(path.read_text())
                if fault=='partial':value['status']='depth_partial'
                elif fault=='duplicate':value['artifacts'][1]['frame_id']='f0'
                elif fault=='component':value['camera_component']='outputs/other'
                else:value['selected_frames'][0]['image_sha256']='0'*64
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):full_surface.prepare_inputs(root,'depth',component,annotations)

    def test_nearest_time_neighbors_and_deterministic_parameter_sample(self):
        frames=[dict(timestamp_seconds=t) for t in (0,.5,1,20,21,22)]
        self.assertEqual(full_surface.nearest_neighbors(frames,2,4),[1,0,3,4])
        values=[np.full(9000,3.,np.float32),np.full(100,3.,np.float32)]
        result=full_surface.scene_parameters(values,100,8)
        self.assertEqual(result['sample_count'],3100)
        self.assertAlmostEqual(result['voxel_length'],.03)
        self.assertAlmostEqual(result['sdf_trunc'],.24)
        self.assertAlmostEqual(result['depth_trunc'],3.3)

    def load_cli(self):
        path=Path(__file__).resolve().parents[1]/'scripts/build_full_surface.py'
        spec=importlib.util.spec_from_file_location('build_full_surface_test',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_cpu_resource_gate_precedes_heavy_imports(self):
        cli=self.load_cli()
        import builtins
        original=builtins.__import__;imports=[]
        def record(name,*args,**kwargs):
            imports.append(name);return original(name,*args,**kwargs)
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(cli.sys,'prefix',str(root/'.venv-geometry')), \
                    patch.object(cli,'cpu_guard',side_effect=ValueError('fresh gate failed')), \
                    patch.object(builtins,'__import__',side_effect=record):
                with self.assertRaisesRegex(ValueError,'fresh gate failed'):
                    cli.execute(root,root,SimpleNamespace())
        self.assertFalse(any(name.startswith(('numpy','torch','open3d')) for name in imports))

    def test_filter_checkpoint_streams_and_preserves_uncertain_coverage(self):
        cli=self.load_cli()
        with TemporaryDirectory() as tmp:
            root=Path(tmp);component,annotations=self.fixture(root)
            inputs=full_surface.prepare_inputs(root,'depth',component,annotations)
            run=root/'outputs/surface';run.mkdir()
            report=cli.filter_frames(inputs,run,lambda index:[],.02)
            self.assertEqual(report['frame_count'],3)
            self.assertTrue((run/'filtered/status.json').is_file())
            for row in report['frames']:
                with np.load(run/row['file'],allow_pickle=False) as mask:
                    self.assertFalse(np.any(mask['strict_mask'] & ~mask['coverage_mask']))
                    self.assertFalse(np.any(mask['coverage_mask'] & ~mask['base_mask']))
                    self.assertEqual(int(mask['uncertain_mask'].sum()),row['counts']['uncertain'])
            self.assertGreater(sum(r['counts']['uncertain'] for r in report['frames']),0)

    def test_cli_failure_persists_status_and_run_collision_preserves_bytes(self):
        cli=self.load_cli()
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            args=['--depth-run','depth','--camera-component','outputs/camera/components/component_000','--run-id','surface']
            with patch.object(cli,'ROOT',root),patch.object(cli,'execute',side_effect=ValueError('resource gate failed')):
                self.assertEqual(cli.main(args),2)
                path=root/'outputs/surface/status.json';before=path.read_bytes()
                self.assertEqual(cli.main(args),2)
                self.assertEqual(path.read_bytes(),before)
                self.assertFalse(json.loads(before)['mesh_ready'])

    def test_execute_preserves_immutable_reports_and_writes_parameters_once(self):
        """Exercise real JSON exclusive-create writes across the complete control flow."""
        cli=self.load_cli()
        import builtins
        from room_reconstruction import meshing
        original_import=builtins.__import__
        def backend_import(name,*args,**kwargs):
            if name=='open3d':return SimpleNamespace(__version__='0.19.0')
            return original_import(name,*args,**kwargs)
        def integrate(frames,**settings):
            self.assertEqual(len(list(frames)),3)
            return SimpleNamespace(vertices=np.array([[0,0,0],[1,0,0],[0,1,0]]),triangles=np.array([[0,1,2]]))
        def export(mesh,run,name):
            artifacts=[]
            for suffix in ('ply','glb'):
                path=run/f'{name}.{suffix}';path.write_bytes(b'backend export stub')
                artifacts.append(dict(file=path.name,sha256=sha256_file(path)))
            return dict(artifacts=artifacts,vertex_count=3,triangle_count=1)
        with TemporaryDirectory() as tmp:
            root=Path(tmp);component,annotations=self.fixture(root)
            source_root=Path(__file__).resolve().parents[1]
            for relative in ('scripts/build_full_surface.py','scripts/fuse_tsdf.py','src/room_reconstruction/full_surface.py',
                    'src/room_reconstruction/full_take.py','src/room_reconstruction/depth_da3.py',
                    'src/room_reconstruction/filtering.py','src/room_reconstruction/semantic_masks.py',
                    'src/room_reconstruction/meshing.py','src/room_reconstruction/camera_geometry.py'):
                destination=root/relative;destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source_root/relative,destination)
            weights=root/'models/masking/ssdlite320_mobilenet_v3_large_coco-a79551df.pth'
            weights.parent.mkdir(parents=True);weights.write_bytes(b'local test checkpoint')
            run=root/'outputs/surface';run.mkdir()
            args=SimpleNamespace(depth_run='depth',camera_component=component,annotations=str(annotations.relative_to(root)),
                confidence_quantile=.02,voxel_divisor=100,sdf_voxels=8,skip_strict=False,run_id='surface')
            with patch.object(cli.sys,'prefix',str(root/'.venv-geometry')), \
                    patch.object(cli,'cpu_guard',return_value=dict(resource_ready=True)), \
                    patch.object(cli,'WEIGHTS_SHA256',sha256_file(weights)), \
                    patch.object(cli,'detector_callback',return_value=lambda index:[]), \
                    patch.object(cli.importlib.metadata,'version',side_effect=lambda name:'0.19.0' if name=='open3d' else 'test'), \
                    patch.object(np,'__version__','1.26.4'), \
                    patch.object(meshing,'integrate_mesh',side_effect=integrate), \
                    patch.object(cli,'export_mesh',side_effect=export), \
                    patch.object(builtins,'__import__',side_effect=backend_import):
                report=cli.execute(root,run,args)
            self.assertEqual(report['status'],'mesh_passed')
            self.assertTrue((run/'parameters.json').is_file())
            self.assertTrue((run/'filtered/integration_report.json').is_file())
            immutable=json.loads((run/'filtered/report.json').read_text())
            self.assertNotIn('integration_parameters',immutable)
            self.assertNotIn('parameters',json.loads((run/'manifest.json').read_text()))
            self.assertGreater(report['counts']['coverage_integrated'],0)


if __name__=='__main__':unittest.main()
