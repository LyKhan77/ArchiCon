import importlib.util
import json
import struct
import unittest
import numpy as np
from room_reconstruction.mesh_audit import pinhole_rays, resolve_hits, boundary_edges, double_sided_glb


class MeshAuditTests(unittest.TestCase):
    def test_rays_preserve_optical_z_and_rigid_camera(self):
        K=np.array([[2.,0,1],[0,2.,1],[0,0,1]])
        pose=np.eye(4);pose[:3,3]=[3,4,5]
        rays=pinhole_rays(K,pose,3,3)
        np.testing.assert_allclose(rays[1,1],[3,4,5,0,0,1])
        np.testing.assert_allclose(rays[0,0,3:],[-.5,-.5,1])

    def geometry(self):
        # Near triangle faces away, farther triangle faces camera.
        v=np.array([[-1,-1,2],[1,-1,2],[0,1,2],[-1,-1,3],[0,1,3],[1,-1,3]],np.float32)
        t=np.array([[0,1,2],[3,4,5]],np.uint32);c=np.array([[1,0,0]]*3+[[0,1,0]]*3,np.float32)
        return v,t,c

    def test_single_side_skips_backface_and_finds_next_front_surface(self):
        v,t,c=self.geometry();rays=np.array([[0,0,0,0,0,1],[3,3,0,0,0,1]],np.float32)
        hits={'ray_ids':np.array([0,0]),'t_hit':np.array([2.,3.]),'primitive_ids':np.array([0,1]),'primitive_uvs':np.array([[.25,.5],[.5,.25]])}
        two=resolve_hits(rays,hits,v,t,c,True);one=resolve_hits(rays,hits,v,t,c,False)
        self.assertEqual(float(two['depth'][0]),2);self.assertEqual(float(one['depth'][0]),3)
        np.testing.assert_allclose(two['rgb'][0],[1,0,0]);np.testing.assert_allclose(one['rgb'][0],[0,1,0])
        self.assertFalse(two['hit'][1]);self.assertFalse(one['hit'][1])

    def test_boundary_edges_open_plane_vs_closed_tetrahedron(self):
        b,n=boundary_edges(np.array([[0,1,2],[0,2,3]]));self.assertEqual(len(b),4);self.assertEqual(len(n),0)
        b,n=boundary_edges(np.array([[0,1,2],[0,3,1],[1,3,2],[2,3,0]]));self.assertEqual(len(b),0)

    def test_double_sided_glb_preserves_binary_geometry_and_scene(self):
        document={'asset':{'version':'2.0'},'meshes':[{'primitives':[{'attributes':{'POSITION':0},'indices':1}]}], 'nodes':[{'mesh':0}]}
        raw=json.dumps(document).encode();raw+=b' '*((-len(raw))%4);binary=b'01234567'
        data=struct.pack('<III',0x46546c67,2,12+8+len(raw)+8+len(binary))+struct.pack('<II',len(raw),0x4e4f534a)+raw+struct.pack('<II',len(binary),0x004e4942)+binary
        changed=double_sided_glb(data);length,kind=struct.unpack('<II',changed[12:20]);j=json.loads(changed[20:20+length])
        self.assertTrue(j['materials'][0]['doubleSided']);self.assertEqual(j['nodes'],document['nodes'])
        self.assertEqual(changed[20+length+8:],binary)

    def test_existing_glb_material_properties_and_indices_are_preserved(self):
        material={'pbrMetallicRoughness':{'baseColorFactor':[.2,.4,.6,1], 'metallicFactor':.3},
            'alphaMode':'BLEND', 'normalTexture':{'index':0}, 'extensions':{'custom':{'value':1}}}
        document={'asset':{'version':'2.0'}, 'materials':[material],
            'meshes':[{'primitives':[{'attributes':{'POSITION':0}, 'material':0}]}]}
        raw=json.dumps(document).encode();raw+=b' '*((-len(raw))%4)
        data=struct.pack('<III',0x46546c67,2,20+len(raw))+struct.pack('<II',len(raw),0x4e4f534a)+raw
        changed=double_sided_glb(data);length=struct.unpack_from('<I',changed,12)[0]
        doc=json.loads(changed[20:20+length]);self.assertTrue(doc['materials'][0].pop('doubleSided'))
        self.assertEqual(doc,document)

    @unittest.skipUnless(importlib.util.find_spec('open3d'),'Raycasting backend tested on server')
    def test_actual_raycast_hits_and_interpolated_rgb(self):
        from room_reconstruction.mesh_audit import TriangleRenderer
        v,t,c=self.geometry();renderer=TriangleRenderer(v,t,c)
        rays=np.array([[[0,0,0,0,0,1]]],np.float32)
        two,one=renderer.render(rays)
        self.assertAlmostEqual(float(two['depth'][0,0]),2,places=5)
        self.assertAlmostEqual(float(one['depth'][0,0]),3,places=5)


if __name__=='__main__':unittest.main()
