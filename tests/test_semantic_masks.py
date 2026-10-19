import unittest
import numpy as np
from room_reconstruction.semantic_masks import annotation_mask, person_boxes


class SemanticMaskTests(unittest.TestCase):
    def test_boxes_select_only_people_above_threshold_and_clamp_padding(self):
        boxes = person_boxes([[0, 10, 20, 40], [0, 0, 100, 100], [5, 5, 8, 8]],
                             [1, 3, 1], [.9, .99, .1], (100, 100))
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0][0], 0)
        self.assertGreater(boxes[0][2], .2)

    def test_polygon_and_person_regions_union_on_exact_grid(self):
        row = {'people_boxes': [[0, 0, .25, .5]],
               'glass_polygons': [[[.5, .5], [1, .5], [1, 1], [.5, 1]]]}
        masks = annotation_mask((8, 8), row)
        self.assertTrue(masks['people'][1, 1]); self.assertFalse(masks['people'][6, 6])
        self.assertTrue(masks['glass'][6, 6]); self.assertFalse(masks['excluded'][2, 6])
        self.assertTrue(np.array_equal(masks['excluded'], masks['people'] | masks['glass']))

    def test_empty_annotations_do_not_drop_low_texture_walls(self):
        self.assertFalse(annotation_mask((8, 9), {})['excluded'].any())

    def test_invalid_annotation_coordinates_rejected(self):
        for row in [{'people_boxes': [[-.1, 0, 1, 1]]},
                    {'glass_polygons': [[[0, 0], [1, 1]]]},
                    {'people_boxes': [[.8, 0, .2, 1]]}]:
            with self.subTest(row=row), self.assertRaises(ValueError): annotation_mask((8, 9), row)


if __name__ == '__main__': unittest.main()
