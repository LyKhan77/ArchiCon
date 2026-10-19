import unittest

import numpy as np

from room_reconstruction import camera_geometry as geometry


class GeometryTests(unittest.TestCase):
    def test_world_camera_inverse_preserves_center_and_rotation(self):
        world_to_camera = np.array([[0, -1, 0, 2], [1, 0, 0, 3], [0, 0, 1, 4]], float)
        pose = geometry.c2w_from_w2c(world_to_camera)
        np.testing.assert_allclose(np.vstack([world_to_camera, [0, 0, 0, 1]]) @ pose, np.eye(4))
        np.testing.assert_allclose(pose[:3, 3], [-3, 2, -4])

    def test_depth_roundtrip_and_resize_use_z_not_ray_length(self):
        k = np.array([[800, 0, 540], [0, 900, 960], [0, 0, 1]], float)
        uv = np.array([[540, 960], [700, 1200]], float)
        xyz = geometry.backproject_zdepth(uv, np.array([2, 3]), k)
        np.testing.assert_allclose(xyz[:, 2], [2, 3])
        np.testing.assert_allclose(geometry.project(xyz, k), uv)
        small = geometry.resize_intrinsics(k, (1080, 1920), (540, 960))
        np.testing.assert_allclose(geometry.project(xyz, small), uv / 2)
        np.testing.assert_array_equal(k[0], [800, 0, 540])

    def test_invalid_geometry_fails_closed(self):
        for pose in [np.diag([-1., 1, 1, 1]), np.full((4, 4), np.nan), np.diag([2., 1, 1, 1])]:
            with self.subTest(pose=pose), self.assertRaises(ValueError):
                geometry.validate_pose(pose)
        with self.assertRaises(ValueError):
            geometry.backproject_zdepth([[0, 0]], [-1], np.eye(3))
        with self.assertRaises(ValueError):
            geometry.project([[1, 2, 0]], np.eye(3))


if __name__ == '__main__':
    unittest.main()
