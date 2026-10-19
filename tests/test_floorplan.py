import unittest
import numpy as np

from room_reconstruction.floorplan import camera_records, section_segments, metric_scale, estimate_floor


class FloorplanTests(unittest.TestCase):
    def test_glb_material_vertex_colors_survive_scene_transform(self):
        try:
            import trimesh
        except ImportError:
            self.skipTest('Trimesh geometry backend is server-only')
        import io
        from room_reconstruction import floorplan
        colors=np.array([[255,0,0,255],[0,255,0,255],[0,0,255,255]],np.uint8)
        mesh=trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]],faces=[[0,1,2]],vertex_colors=colors,process=False)
        # A material makes the GLB importer use TextureVisuals with a separate COLOR_0 attribute.
        from room_reconstruction.mesh_audit import double_sided_glb
        data=double_sided_glb(mesh.export(file_type='glb'))
        scene=trimesh.load(io.BytesIO(data),file_type='glb',force='scene',process=False)
        scene.apply_translation([2,3,4])
        result=floorplan.colored_scene_mesh(scene)
        np.testing.assert_array_equal(result.visual.vertex_colors,colors)
        np.testing.assert_allclose(result.vertices[0],[2,3,4])

    def test_wall_slice_is_horizontal_and_horizontal_surfaces_excluded(self):
        vertices = np.array([[0,0,0],[2,0,0],[2,2,0],[0,2,0], [0,1,0],[2,1,0],[2,1,2]],float)
        faces = np.array([[0,1,2],[0,2,3],[4,5,6]])
        result = section_segments(vertices, faces, 1)
        self.assertEqual(result.shape,(2,2,2))
        self.assertAlmostEqual(np.linalg.norm(result[:,1]-result[:,0],axis=1).sum(),2)
        np.testing.assert_allclose(result[:,:,1],0)

    def test_viewpoints_preserve_forward_direction_and_break_temporal_gaps(self):
        T=np.eye(4);T[:3,3]=[3,4,5]
        frames=[dict(frame_id=str(i),timestamp_seconds=t,c2w=np.eye(4).tolist()) for i,t in enumerate([0,1,4])]
        cameras,path=camera_records(frames,T)
        pose=np.array(cameras[0]['matrix'])
        np.testing.assert_allclose(-pose[:3,2],[0,0,1])
        np.testing.assert_allclose(pose[:3,3],[3,4,5])
        self.assertEqual([p['break_before'] for p in path],[True,False,True])
        frames[0]['K']=[[500,0,250],[0,500,500],[0,0,1]]
        frames[0]['image_size']=[500,1000]
        self.assertAlmostEqual(camera_records(frames,T)[0][0]['vertical_fov_degrees'],90)
        bad=T.copy();bad[0,0]=2
        with self.assertRaises(ValueError):camera_records(frames,bad)

    def test_meter_calibration_requires_positive_finite_reference(self):
        self.assertAlmostEqual(metric_scale([0,0],[3,4],2.5),.5)
        for true_length in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):metric_scale([0,0],[1,0],true_length)
        with self.assertRaises(ValueError):metric_scale([1,1],[1,1],2)
        with self.assertRaises(ValueError):metric_scale([0,0],[.1,0],1e308)

    def test_floor_estimate_uses_area_below_camera_not_ceiling(self):
        v=np.array([[0,0,0],[2,0,0],[2,0,2],[0,0,2],[0,3,0],[2,3,0],[2,3,2]],float)
        f=np.array([[0,1,2],[0,2,3],[4,5,6]])
        self.assertAlmostEqual(estimate_floor(v,f,np.array([1.5,1.6])),0)


if __name__=='__main__':unittest.main()
