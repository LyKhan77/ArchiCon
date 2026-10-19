import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np


def cli():
    path=Path(__file__).resolve().parents[1]/'scripts/audit_mesh_holes.py'
    spec=importlib.util.spec_from_file_location('audit_cli_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class FixedViewTests(unittest.TestCase):
    def fixture(self,root):
        module=cli()
        def write(name,value):
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value));return path
        camera=write('outputs/l2bp3_camera_pilot_002/undistorted/cameras.json',{'frames':[]})
        map_path=write('outputs/l2bp3_camera_pilot_002/primary/images.bin',{'fixed':'map'})
        map_hashes={'primary/images.bin':module.fusion.sha256_file(map_path)}
        prefix='outputs/l2bp3_validation_opencv_002/'
        write(prefix+'status.json',{'status':'localization_passed'})
        write(prefix+'manifest.json',{'stage':'validation_localization',
            'camera_run':'l2bp3_camera_pilot_002','map_hashes_before':map_hashes})
        write(prefix+'report.json',{'stage':'validation_localization','map_unchanged':True,
            'calibration_fixed':True,'training_map_fixed':True,'bundle_adjustment_run':False,
            'scene_optimizer_run':False,'query_triangulation_run':False,
            'map_hashes_before':map_hashes,'map_hashes_after':map_hashes})
        frames=[]
        for index in range(8):
            image=write(prefix+f'undistorted/images/{index}.jpg',{'image':index})
            frames.append({'frame_id':f'f{index}','split':'validation','image_name':image.name,
                'image_sha256':module.fusion.sha256_file(image),'K':[[400,0,100],[0,400,200],[0,0,1]],
                'image_size':[200,400],'c2w':np.eye(4).tolist()})
        doc={'frames':frames,'image_root':'images','calibration_fixed':True,'training_map_fixed':True,
            'calibration_reference':str(camera),'calibration_reference_sha256':module.fusion.sha256_file(camera)}
        path=write(prefix+'undistorted/cameras.json',doc)
        write('outputs/l2bp3_tsdf_001/reports/viewer_export.json',{'world_to_viewer':np.eye(4).tolist()})
        reference={'meshes':{'filtered':{'bounds':[[-2,-1,-3],[2,1,3]]}}}
        return module,path,doc,map_path,reference

    def test_fixed_view_resize_and_proper_exterior_camera(self):
        with TemporaryDirectory() as temp:
            root=Path(temp).resolve();module,path,doc,map_path,reference=self.fixture(root)
            views=module.make_views(root,reference,{})
            self.assertEqual(len(views),9)
            np.testing.assert_allclose(views[0]['K'],[[560,0,140],[0,504,252],[0,0,1]])
            rotation=np.asarray(views[-1]['c2w'])[:3,:3]
            np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-10)
            self.assertAlmostEqual(np.linalg.det(rotation),1)

    def test_rejects_changed_calibration_map_or_localization_world(self):
        for fault in ('calibration','map','world','optimization'):
            with self.subTest(fault=fault), TemporaryDirectory() as temp:
                root=Path(temp).resolve();module,path,doc,map_path,reference=self.fixture(root)
                if fault=='calibration':
                    doc['calibration_reference_sha256']='changed';path.write_text(json.dumps(doc))
                if fault=='map':map_path.write_bytes(b'changed map')
                if fault=='world':
                    p=root/'outputs/l2bp3_validation_opencv_002/manifest.json';m=json.loads(p.read_text())
                    m['camera_run']='different_camera_world';p.write_text(json.dumps(m))
                if fault=='optimization':
                    p=root/'outputs/l2bp3_validation_opencv_002/report.json';m=json.loads(p.read_text())
                    m['scene_optimizer_run']=True;p.write_text(json.dumps(m))
                with self.assertRaises(ValueError):module.make_views(root,reference,{})


if __name__=='__main__':unittest.main()
