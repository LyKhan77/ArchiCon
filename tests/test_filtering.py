import unittest
import numpy as np
from room_reconstruction.filtering import spatial_masks, view_evidence, multiview_mask, depth_normals


class FilteringTests(unittest.TestCase):
    def frame(self, depth):
        return {'depth': np.full((7, 9), depth, dtype=np.float32),
                'K': np.array([[12., 0, 4], [0, 12., 3], [0, 0, 1]]),
                'w2c': np.eye(4), 'valid': np.ones((7, 9), bool)}

    def test_spatial_retains_smooth_low_texture_plane_and_removes_semantic(self):
        d = self.frame(2)['depth']; semantic = np.zeros_like(d, bool); semantic[3, 4] = True
        masks = spatial_masks(d, np.full_like(d, 5), semantic)
        self.assertEqual(int(masks['base'].sum()), d.size - 1)
        self.assertFalse(masks['base'][3, 4])

    def test_nonfinite_invalid_depth_and_discontinuity_are_removed(self):
        d = self.frame(2)['depth']; d[:, 5:] = 4; d[0, 0] = np.nan; d[0, 1] = -1
        masks = spatial_masks(d, np.ones_like(d)*5, np.zeros_like(d, bool))
        self.assertFalse(masks['valid'][0, 0]); self.assertFalse(masks['valid'][0, 1])
        self.assertFalse(masks['base'][3, 4]); self.assertFalse(masks['base'][3, 5])
        self.assertTrue(masks['base'][3, 1])

    def test_equal_surface_supports_and_out_of_view_is_unknown(self):
        a, b = self.frame(2), self.frame(2); b['w2c'][0, 3] = -.5
        evidence = view_evidence(a, b)
        self.assertEqual(int(evidence['contradiction'].sum()), 0)
        self.assertGreater(int(evidence['support'].sum()), 0)
        self.assertGreater(int(evidence['unknown'].sum()), 0)

    def test_occluded_surface_is_not_a_contradiction(self):
        evidence = view_evidence(self.frame(4), self.frame(2))
        self.assertTrue(evidence['occluded'].all())
        self.assertFalse(evidence['contradiction'].any())

    def test_subpixel_projection_outside_image_remains_unknown(self):
        a, b = self.frame(2), self.frame(2)
        b['w2c'][0, 3] = -.2*2/12
        evidence = view_evidence(a, b)
        self.assertTrue(evidence['unknown'][:, 0].all())
        self.assertFalse(evidence['support'][:, 0].any())

    def test_surface_in_front_of_target_is_contradiction(self):
        evidence = view_evidence(self.frame(2), self.frame(4))
        self.assertTrue(evidence['contradiction'].all())
        self.assertFalse(evidence['occluded'].any())

    def test_masked_target_is_unknown_and_support_count_gate_is_explicit(self):
        a, b = self.frame(2), self.frame(2); b['valid'][:] = False
        result = multiview_mask(a, [b])
        self.assertFalse(result['mask'].any())
        self.assertFalse(result['contradictions'].any())
        self.assertTrue(result['unsupported'].all())
        result = multiview_mask(a, [self.frame(2), self.frame(2)], min_support=2)
        self.assertTrue(result['mask'].all())

    def test_depth_normals_face_camera_with_unit_length(self):
        a = self.frame(2); normals, valid = depth_normals(a['depth'], a['K'], a['valid'])
        np.testing.assert_allclose(normals[valid], np.tile([0, 0, -1], (valid.sum(), 1)), atol=1e-6)
        self.assertFalse(valid[0].any())

    def test_invalid_thresholds_and_shapes_rejected(self):
        a = self.frame(2)
        with self.assertRaises(ValueError): view_evidence(a, a, relative_tolerance=-.1)
        with self.assertRaises(ValueError): spatial_masks(a['depth'], np.zeros((2, 2)), a['valid'])
        with self.assertRaises(ValueError): multiview_mask(a, [a], min_support=0)


if __name__ == '__main__': unittest.main()
