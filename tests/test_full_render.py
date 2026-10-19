import importlib.util
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
import numpy as np


def cli():
    spec=importlib.util.spec_from_file_location('full_render_test',Path(__file__).resolve().parents[1]/'scripts/render_full_surface.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class FullRenderTests(unittest.TestCase):
    def test_viewer_basis_maps_estimated_up_without_changing_scale(self):
        module=cli();poses=np.repeat(np.eye(4)[None],5,axis=0)
        poses[:,:3,3]=[[i,0,i*.5] for i in range(5)]
        bounds=np.array([[-2,-1,-2],[4,3,8]])
        T=module.display_transform(poses,bounds);R=T[:3,:3]
        np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-7)
        self.assertAlmostEqual(np.linalg.det(R),1)
        np.testing.assert_allclose(R@np.array([0,-1,0]),[0,1,0],atol=1e-7)
        np.testing.assert_allclose(R@bounds.mean(0)+T[:3,3],[0,0,0],atol=1e-7)

    def test_glb_root_transform_and_material_preserve_binary_geometry(self):
        doc={'asset':{'version':'2.0'},'nodes':[{'mesh':0}], 'scenes':[{'nodes':[0]}], 'scene':0,
            'meshes':[{'primitives':[{'attributes':{'POSITION':0}}]}]}
        raw=json.dumps(doc).encode();raw+=b' '*((-len(raw))%4);binary=b'unchanged buffer'
        binary+=b'\x00'*((-len(binary))%4)
        body=struct.pack('<II',len(raw),0x4e4f534a)+raw+struct.pack('<II',len(binary),0x004e4942)+binary
        data=struct.pack('<III',0x46546c67,2,len(body)+12)+body
        with TemporaryDirectory() as temp:
            source=Path(temp)/'source.glb';dest=Path(temp)/'review.glb';source.write_bytes(data)
            T=np.eye(4);T[:3,3]=[2,3,4];cli().review_glb(source,dest,T)
            out=dest.read_bytes();size=struct.unpack_from('<I',out,12)[0];changed=json.loads(out[20:20+size])
            self.assertEqual(out[20+size+8:],binary)
            self.assertEqual(changed['nodes'][0],doc['nodes'][0])
            self.assertEqual(changed['nodes'][1]['children'],[0])
            self.assertEqual(changed['nodes'][1]['matrix'],T.T.ravel().tolist())
            self.assertTrue(changed['materials'][0]['doubleSided'])
            with self.assertRaises(FileExistsError):cli().review_glb(source,dest,T)

    def test_vertical_or_stationary_camera_path_has_proper_display_basis(self):
        for canceled in (False,True):
            with self.subTest(canceled=canceled):
                poses=np.repeat(np.eye(4)[None],4,axis=0)
                poses[:,:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
                poses[:,:3,3]=[[i,0,0] for i in range(4)]
                if canceled:poses[2:,:3,:3]=np.diag([-1,-1,1])@poses[2:,:3,:3]
                T=cli().display_transform(poses,np.array([[-1,-1,-1],[1,1,1]]))
                self.assertTrue(np.isfinite(T).all());self.assertAlmostEqual(np.linalg.det(T[:3,:3]),1)


if __name__=='__main__':unittest.main()
