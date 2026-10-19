import importlib.util
import unittest
import numpy as np
from room_reconstruction.meshing import integration_arrays, integrate_mesh


class MeshTests(unittest.TestCase):
    def frame(self):
        return {'depth': np.full((48, 64), 2., np.float32),
                'processed_rgb': np.full((48, 64, 3), 180, np.uint8),
                'K': np.array([[55., 0, 32], [0, 55., 24], [0, 0, 1]]),
                'w2c': np.eye(4)}

    def test_float_z_units_preserved_and_masked_depth_becomes_zero(self):
        f=self.frame();f['depth'][0,0]=np.nan;m=np.ones(f['depth'].shape,bool);m[3,4]=False
        d,rgb,K,E=integration_arrays(f,m,5)
        self.assertEqual(d.dtype,np.float32);self.assertEqual(float(d[2,2]),2.)
        self.assertEqual(float(d[0,0]),0);self.assertEqual(float(d[3,4]),0)
        np.testing.assert_array_equal(E,np.eye(4))

    def test_bad_rgb_grid_or_transform_rejected(self):
        f=self.frame();f['processed_rgb']=np.ones((4,4,3),np.uint8)
        with self.assertRaises(ValueError):integration_arrays(f,np.ones((48,64),bool),5)
        f=self.frame();f['w2c'][0,0]=2
        with self.assertRaises(ValueError):integration_arrays(f,np.ones((48,64),bool),5)

    @unittest.skipUnless(importlib.util.find_spec('open3d'),'Open3D integration runs on server')
    def test_real_tsdf_recovers_world_plane_with_nonidentity_camera(self):
        frames=[]
        for x in (-.12,0,.12):
            f=self.frame();f['w2c'][0,3]=-x
            frames.append((f,np.ones(f['depth'].shape,bool)))
        mesh=integrate_mesh(frames,voxel_length=.035,sdf_trunc=.14,depth_trunc=4)
        xyz=np.asarray(mesh.vertices)
        self.assertGreater(len(mesh.triangles),100)
        self.assertLess(float(np.median(np.abs(xyz[:,2]-2))),.04)
        self.assertLess(abs(float(np.median(xyz[:,0]))),.15)


if __name__=='__main__':unittest.main()
