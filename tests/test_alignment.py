import unittest

import numpy as np

from room_reconstruction import alignment


def transform(scale=2., angle=.4, translation=(3., -2., 1.)):
    c, s = np.cos(angle), np.sin(angle)
    result = np.eye(4)
    result[:3, :3] = scale * np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    result[:3, 3] = translation
    return result


def cameras(points, matrix=None, prefix='f'):
    matrix = np.eye(4) if matrix is None else matrix
    rotation = matrix[:3, :3] / np.cbrt(np.linalg.det(matrix[:3, :3]))
    centers = alignment.transform_points(points, matrix)
    result = []
    for index, center in enumerate(centers):
        pose = np.eye(4); pose[:3, :3] = rotation; pose[:3, 3] = center
        result.append(dict(frame_id=f'{prefix}{index}', timestamp_seconds=float(index), c2w=pose.tolist()))
    return result


class SimilarityTests(unittest.TestCase):
    def setUp(self):
        self.points = np.random.default_rng(23).normal(size=(12, 3))

    def test_recovers_scale_rotation_translation(self):
        expected = transform()
        target = alignment.transform_points(self.points, expected)
        np.testing.assert_allclose(alignment.fit_similarity(self.points, target), expected, atol=1e-12)

    def test_rejects_insufficient_collinear_nonfinite_and_reflected_inputs(self):
        for source, target in [
            (self.points[:2], self.points[:2]),
            (np.arange(12).reshape(4, 3), np.arange(12).reshape(4, 3)),
            (self.points * np.nan, self.points),
            (self.points, self.points * [-1, 1, 1]),
        ]:
            with self.subTest(source=source), self.assertRaises(ValueError):
                alignment.fit_similarity(source, target)

    def test_edge_uses_disjoint_validation_frames_and_camera_orientation(self):
        target = cameras(self.points, transform())
        edge = alignment.evaluate_edge('a', cameras(self.points), 'b', target)
        self.assertTrue(edge['accepted'], edge)
        self.assertFalse(set(edge['fit_frame_ids']) & set(edge['validation_frame_ids']))
        self.assertGreaterEqual(len(edge['validation_frame_ids']), 2)
        # A corrupt held-out camera must not be absorbed into the fit.
        bad = [dict(row) for row in target]
        index = next(i for i, row in enumerate(bad) if row['frame_id'] in edge['validation_frame_ids'])
        bad[index] = dict(bad[index], c2w=np.eye(4).tolist())
        self.assertFalse(alignment.evaluate_edge('a', cameras(self.points), 'b', bad)['accepted'])
        wrong_orientation = cameras(self.points, transform())
        wrong_orientation[0]['c2w'] = np.asarray(wrong_orientation[0]['c2w'])
        wrong_orientation[0]['c2w'][:3, :3] = np.eye(3)
        self.assertFalse(alignment.evaluate_edge('a', cameras(self.points), 'b', wrong_orientation)['accepted'])

    def test_small_overlap_is_reported_without_acceptance(self):
        edge = alignment.evaluate_edge('a', cameras(self.points[:4]), 'b', cameras(self.points[:4]))
        self.assertFalse(edge['accepted'])
        self.assertEqual(edge['shared_count'], 4)
        self.assertIn('shared', edge['reason'])

    def test_duplicate_frame_ids_fail_closed(self):
        rows = cameras(self.points)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            alignment.build_alignment(dict(main=rows + rows[:1], other=rows))

    def test_graph_composes_transforms_and_preserves_disconnected_world(self):
        first, second = transform(), transform(.7, -.2, (8, 0, 2))
        bridge = cameras(self.points, prefix='g')
        middle = cameras(self.points, first) + bridge
        end = cameras(self.points, second, prefix='g')
        data = dict(main=cameras(self.points), b=middle, c=end, isolated=cameras(self.points, prefix='x'))
        report = alignment.build_alignment(data)
        self.assertEqual(report['aligned_components'], ['main', 'b', 'c'])
        self.assertEqual(report['fallback_components'], ['isolated'])
        np.testing.assert_allclose(report['component_to_root']['b'], np.linalg.inv(first), atol=1e-12)
        np.testing.assert_allclose(report['component_to_root']['c'], np.linalg.inv(first) @ np.linalg.inv(second), atol=1e-12)
        reverse = alignment.build_alignment(dict(reversed(list(data.items()))))
        self.assertEqual(report, reverse)

    def test_no_alignment_leaves_all_worlds_separate(self):
        report = alignment.build_alignment(dict(main=cameras(self.points), b=cameras(self.points, prefix='x')))
        self.assertEqual(report['status'], 'no_alignment')
        self.assertEqual(report['fallback_components'], ['b'])

    def test_inconsistent_cycle_rejects_the_connected_candidate(self):
        root = cameras(self.points, prefix='f') + cameras(self.points, prefix='g')
        b = cameras(self.points, prefix='f') + cameras(self.points, prefix='h')
        c = cameras(self.points, prefix='g') + cameras(self.points, transform(1, 0, (10, 0, 0)), prefix='h')
        report = alignment.build_alignment(dict(main=root, b=b, c=c))
        self.assertEqual(report['status'], 'no_alignment')
        self.assertTrue(report['cycle_conflicts'])
        self.assertEqual(report['fallback_components'], ['b', 'c'])


if __name__ == '__main__':
    unittest.main()
