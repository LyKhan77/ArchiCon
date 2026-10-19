"""Actual triangle visibility, boundary diagnostics, and display-only GLB changes."""
import json
import struct
import numpy as np
from .camera_geometry import validate_pose


def pinhole_rays(K,c2w,width,height):
    K=np.asarray(K,float);pose=validate_pose(c2w)
    if K.shape!=(3,3) or not np.isfinite(K).all() or min(K[0,0],K[1,1])<=0 or min(width,height)<=0 or not np.allclose(K[2],[0,0,1]):
        raise ValueError('Invalid ray camera')
    yy,xx=np.indices((height,width))
    pixels=np.stack([xx,yy,np.ones_like(xx)],axis=-1)
    directions=pixels@np.linalg.inv(K).T@pose[:3,:3].T
    origins=np.broadcast_to(pose[:3,3],directions.shape)
    return np.concatenate([origins,directions],axis=-1).astype(np.float32)


def resolve_hits(rays,hits,vertices,triangles,colors,double_sided=True):
    rays=np.asarray(rays,np.float32).reshape(-1,6)
    ids=np.asarray(hits['ray_ids'],np.int64);t=np.asarray(hits['t_hit'])
    primitive=np.asarray(hits['primitive_ids'],np.int64);uv=np.asarray(hits['primitive_uvs'])
    depth=np.full(len(rays),np.inf,np.float32);rgb=np.zeros((len(rays),3),np.float32)
    if not len(ids):return {'depth':depth,'rgb':rgb,'hit':np.isfinite(depth)}
    if (ids.shape!=t.shape or primitive.shape!=t.shape or uv.shape!=(len(t),2)
            or (ids<0).any() or (ids>=len(rays)).any() or (primitive<0).any() or (primitive>=len(triangles)).any()):
        raise ValueError('Invalid triangle intersections')
    valid=np.isfinite(t)&(t>1e-6)&np.isfinite(uv).all(1)
    if not double_sided:
        faces=vertices[triangles[primitive]]
        normals=np.cross(faces[:,1]-faces[:,0],faces[:,2]-faces[:,0])
        valid &= np.sum(normals*rays[ids,3:],axis=1)<0
    indices=np.flatnonzero(valid)
    if len(indices):
        order=indices[np.lexsort((t[indices],ids[indices]))]
        selected=order[np.r_[True,np.diff(ids[order])!=0]]
        rid=ids[selected];weights=np.column_stack([1-uv[selected].sum(1),uv[selected]])
        rgb[rid]=np.sum(colors[triangles[primitive[selected]]]*weights[...,None],axis=1)
        depth[rid]=t[selected]
    return {'depth':depth,'rgb':np.clip(rgb,0,1),'hit':np.isfinite(depth)}


class TriangleRenderer:
    def __init__(self,vertices,triangles,colors,threads=4):
        import open3d as o
        self.o=o;self.threads=threads
        self.vertices=np.ascontiguousarray(vertices,dtype=np.float32)
        self.triangles=np.ascontiguousarray(triangles,dtype=np.uint32)
        self.colors=np.ascontiguousarray(colors,dtype=np.float32)
        if (self.vertices.ndim!=2 or self.vertices.shape[1]!=3 or self.colors.shape!=self.vertices.shape
                or self.triangles.ndim!=2 or self.triangles.shape[1]!=3 or not len(self.triangles)
                or int(self.triangles.max())>=len(self.vertices) or not np.isfinite(self.vertices).all()
                or not np.isfinite(self.colors).all() or not 1<=threads<=4):
            raise ValueError('Invalid colored triangle surface')
        self.scene=o.t.geometry.RaycastingScene(nthreads=threads)
        self.scene.add_triangles(o.core.Tensor(self.vertices),o.core.Tensor(self.triangles))

    def render(self,rays,chunk_size=32768):
        shape=np.asarray(rays).shape[:-1];flat=np.asarray(rays,np.float32).reshape(-1,6)
        outputs=[{'depth':np.full(len(flat),np.inf,np.float32),'rgb':np.zeros((len(flat),3),np.float32),'hit':np.zeros(len(flat),bool)} for _ in range(2)]
        for start in range(0,len(flat),chunk_size):
            block=flat[start:start+chunk_size]
            hit=self.scene.list_intersections(self.o.core.Tensor(block),nthreads=self.threads)
            hits={k:v.numpy() for k,v in hit.items()}
            for output,sides in zip(outputs,(True,False),strict=True):
                rendered=resolve_hits(block,hits,self.vertices,self.triangles,self.colors,sides)
                for k,v in rendered.items():output[k][start:start+len(block)]=v
        for output in outputs:
            for k in output:output[k]=output[k].reshape((*shape,3) if k=='rgb' else shape)
        return tuple(outputs)


def boundary_edges(triangles):
    t=np.asarray(triangles,np.int64)
    if t.ndim!=2 or t.shape[1]!=3 or (t<0).any():raise ValueError('Triangle indices required')
    if not len(t):return np.empty((0,2),np.int64),np.empty((0,2),np.int64)
    stride=int(t.max())+1
    keys=[]
    for a,b in ((0,1),(1,2),(2,0)):
        lo=np.minimum(t[:,a],t[:,b]);hi=np.maximum(t[:,a],t[:,b]);keys.append(lo*stride+hi)
    unique,counts=np.unique(np.concatenate(keys),return_counts=True)
    def decode(values):return np.column_stack([values//stride,values%stride])
    return decode(unique[counts==1]),decode(unique[counts>2])


def double_sided_glb(data):
    if len(data)<20:raise ValueError('Truncated GLB')
    magic,version,total=struct.unpack_from('<III',data)
    if magic!=0x46546c67 or version!=2 or total!=len(data):raise ValueError('Expected GLB2')
    chunks=[];offset=12
    while offset<len(data):
        if offset+8>len(data):raise ValueError('Truncated GLB chunk header')
        length,kind=struct.unpack_from('<II',data,offset);offset+=8
        if length%4 or offset+length>len(data):raise ValueError('Malformed GLB chunk')
        chunks.append((kind,data[offset:offset+length]));offset+=length
    if not chunks or chunks[0][0]!=0x4e4f534a:raise ValueError('Missing GLB JSON')
    doc=json.loads(chunks[0][1]);materials=doc.setdefault('materials',[])
    for material in materials:material['doubleSided']=True
    default=None
    for mesh in doc.get('meshes',[]):
        for primitive in mesh['primitives']:
            if 'material' not in primitive:
                if default is None:default=len(materials);materials.append({'doubleSided':True})
                primitive['material']=default
    encoded=json.dumps(doc,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%4)
    chunks[0]=(0x4e4f534a,encoded)
    body=b''.join(struct.pack('<II',len(payload),kind)+payload for kind,payload in chunks)
    return struct.pack('<III',magic,version,12+len(body))+body
