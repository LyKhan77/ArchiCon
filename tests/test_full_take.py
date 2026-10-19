import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from room_reconstruction.full_take import plan_windows, prepare_component, validate_prediction, trajectory_outliers


class FullTakeTests(unittest.TestCase):
    def test_isolated_camera_teleport_flagged_after_scale_and_translation(self):
        frames=[]
        for i in range(40):
            pose=np.eye(4);pose[:3,3]=[i*.1,0,np.sin(i*.1)]
            frames.append(dict(frame_id=f'f{i}',timestamp_seconds=i*.5,c2w=pose.tolist()))
        self.assertEqual(trajectory_outliers(frames),[])
        frames[20]['c2w'][0][3]=75000
        self.assertEqual([r['frame_id'] for r in trajectory_outliers(frames)],['f20'])
        for frame in frames:
            pose=np.asarray(frame['c2w']);pose[:3,3]=pose[:3,3]*.01+[3,5,7];frame['c2w']=pose.tolist()
        self.assertEqual([r['frame_id'] for r in trajectory_outliers(frames)],['f20'])

    def test_windows_cover_whole_sequence_and_endpoints_without_repeated_views(self):
        for n in (3,4,15,16,17,28,52,242,302):
            with self.subTest(n=n):
                windows=plan_windows(n)
                self.assertEqual(set().union(*map(set,windows)),set(range(n)))
                self.assertEqual(windows[0][0],0);self.assertEqual(windows[-1][-1],n-1)
                self.assertTrue(all(3<=len(w)<=16 and len(w)==len(set(w)) for w in windows))
                self.assertTrue(all(len(set(a)&set(b))>=4 for a,b in zip(windows,windows[1:])))
        for bad in (0,2,True,3.5):
            with self.assertRaises(ValueError):plan_windows(bad)

    def fixture(self,root):
        from room_reconstruction.provenance import sha256_file
        root=root.resolve();root.mkdir(exist_ok=True)
        images=root/'undistorted/images';images.mkdir(parents=True)
        native=[];frames=[]
        for i in range(4):
            image=images/f'{i}.jpg';image.write_bytes(f'image{i}'.encode())
            pose=np.eye(4);pose[:3,3]=[i,i*i*.1,0]
            source={'frame_id':f'f{i}','source_index':i,'timestamp_seconds':i*.5,'split':'train' if i%2 else 'validation'}
            native.append(source)
            frames.append(dict(source,split='train',audit_split=source['split'],reconstruction_role='production',
                image_name=image.name,image_sha256=sha256_file(image),image_size=[1080,1920],
                camera_model='PINHOLE',K=[[900,0,540],[0,900,960],[0,0,1]],c2w=pose.tolist()))
        files={'manifest':{'stage':'camera_full_component','selected_frames':native},
            'status':{'stage':'camera_full_component','status':'camera_passed','camera_ready':True},
            'report':{'stage':'camera_full_component','quality':{'passed':True},'undistorted':True}}
        for name,data in files.items():(root/f'{name}.json').write_text(json.dumps(data))
        (root/'undistorted/cameras.json').write_text(json.dumps({'schema_version':1,
            'pose_convention':'OpenCV c2w; +x right, +y down, +z forward',
            'image_root':'images','frames':frames}))
        return root,frames

    def test_component_preserves_production_scope_and_checks_images(self):
        with TemporaryDirectory() as tmp:
            root,frames=self.fixture(Path(tmp))
            result=prepare_component(root)
            self.assertEqual(len(result['frames']),4)
            self.assertEqual(result['frames'][0]['audit_split'],'validation')
            (root/'undistorted/images/0.jpg').write_bytes(b'changed')
            with self.assertRaises(ValueError):prepare_component(root)

    def test_wrong_stage_pose_or_original_split_rejected(self):
        for fault in ('stage','pose','split'):
            with self.subTest(fault=fault), TemporaryDirectory() as tmp:
                root,frames=self.fixture(Path(tmp))
                if fault=='stage':
                    p=root/'status.json';d=json.loads(p.read_text());d['stage']='camera_pilot';p.write_text(json.dumps(d))
                else:
                    p=root/'undistorted/cameras.json';d=json.loads(p.read_text())
                    if fault=='pose':d['frames'][0]['c2w'][0][0]=2
                    else:d['frames'][0]['audit_split']='train'
                    p.write_text(json.dumps(d))
                with self.assertRaises(ValueError):prepare_component(root)

    def test_simple_pinhole_undistorter_output_is_accepted(self):
        with TemporaryDirectory() as tmp:
            root,frames=self.fixture(Path(tmp))
            path=root/'undistorted/cameras.json';doc=json.loads(path.read_text())
            for frame in doc['frames']:frame['camera_model']='SIMPLE_PINHOLE'
            path.write_text(json.dumps(doc))
            self.assertEqual(len(prepare_component(root)['frames']),4)

    def test_variable_window_prediction_stays_in_same_global_camera(self):
        from types import SimpleNamespace
        from room_reconstruction.depth_da3 import expected_resize
        n=3;K=np.repeat(np.array([[[900.,0,540],[0,900,960],[0,0,1]]]),n,axis=0)
        poses=np.repeat(np.eye(4)[None],n,axis=0);expected,hw=expected_resize(K,[1080,1920])
        p=SimpleNamespace(depth=np.ones((n,*hw),np.float32),conf=np.ones((n,*hw),np.float32),
            intrinsics=expected,extrinsics=poses[:,:3],processed_images=np.zeros((n,*hw,3),np.uint8),is_metric=False)
        self.assertEqual(validate_prediction(p,K,poses,[1080,1920])['view_count'],3)
        p.extrinsics=p.extrinsics.copy();p.extrinsics[0,0,3]=1
        with self.assertRaises(ValueError):validate_prediction(p,K,poses,[1080,1920])


if __name__=='__main__':unittest.main()
