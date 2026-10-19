import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np

from room_reconstruction.pipeline_cameras import quarantine_extreme_poses
from room_reconstruction.full_take import prepare_component
from room_reconstruction.provenance import sha256_file


class CameraQuarantineTests(unittest.TestCase):
    def test_one_extreme_pose_does_not_discard_the_valid_room_component(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();component=root/'outputs/camera/components/component_000';images=component/'undistorted/images';images.mkdir(parents=True)
            job=root/'outputs/job';job.mkdir()
            frames=[];selected=[]
            for i in range(20):
                image=images/f'{i}.jpg';image.write_bytes(f'actual bytes{i}'.encode())
                pose=np.eye(4);pose[:3,3]=[i*.1,0,0]
                if i==10:pose[0,3]=75000
                identity=dict(frame_id=f'f{i}',source_index=i,timestamp_seconds=i*.5,split='train')
                selected.append(identity)
                frames.append(dict(identity,audit_split='train',reconstruction_role='production',image_name=image.name,
                  image_sha256=sha256_file(image),image_size=[100,200],camera_model='PINHOLE',K=[[90,0,50],[0,90,100],[0,0,1]],c2w=pose.tolist()))
            def write(name,value):(component/name).write_text(json.dumps(value))
            write('status.json',dict(stage='camera_full_component',status='camera_passed',camera_ready=True))
            write('manifest.json',dict(stage='camera_full_component',source_sha256='video',selected_frames=selected))
            write('report.json',dict(stage='camera_full_component',quality={'passed':True},undistorted=True))
            write('undistorted/cameras.json',dict(schema_version=1,pose_convention='OpenCV c2w; +x right, +y down, +z forward',image_root='images',frames=frames))
            before={str(p.relative_to(component)):sha256_file(p) for p in component.rglob('*') if p.is_file()}
            with self.assertRaises(ValueError):prepare_component(component)
            derived,removed=quarantine_extreme_poses(root,component,job)
            self.assertEqual(removed,['f10']);self.assertEqual(len(prepare_component(derived)['frames']),19)
            self.assertEqual(before,{str(p.relative_to(component)):sha256_file(p) for p in component.rglob('*') if p.is_file()})
            self.assertEqual(quarantine_extreme_poses(root,component,job)[0],derived)


if __name__=='__main__':unittest.main()
