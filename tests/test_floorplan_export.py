import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np


class FloorplanExportTests(unittest.TestCase):
    def test_observer_outside_mesh_does_not_clip_geometric_evidence(self):
        spec=importlib.util.spec_from_file_location('floorplan_export',Path(__file__).resolve().parents[1]/'scripts/export_room_tour.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        wall=np.array([[[0,0],[2,0]],[[2,0],[2,2]],[[2,2],[0,2]],[[0,2],[0,0]]],float)
        path=[dict(point=[-4,1],time=0,index=0,break_before=True),dict(point=[-3,1],time=1,index=1,break_before=False)]
        with TemporaryDirectory() as temporary:
            result=module.drawing([wall]*3,[[0,0],[2,2]],path,Path(temporary),'test')
        self.assertEqual(result['bounds'],[[0,0],[2,2]])
        image_lo,image_hi=np.array(result['image_bounds'])
        self.assertTrue((image_lo<=[-4,0]).all())
        self.assertTrue((image_hi>=[2,2]).all())
        self.assertGreaterEqual(len(result['segments']),4)


if __name__=='__main__':unittest.main()
