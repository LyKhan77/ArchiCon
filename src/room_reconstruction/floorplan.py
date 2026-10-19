"""Evidence-based horizontal sections; relative scale until explicit calibration."""
from __future__ import annotations

import numpy as np


def colored_scene_mesh(scene):
    """Apply scene transforms once without Trimesh dropping GLB COLOR_0."""
    import trimesh
    for geometry in scene.geometry.values():
        attributes=getattr(geometry.visual,'vertex_attributes',{})
        if 'color' in attributes:
            geometry.visual=trimesh.visual.ColorVisuals(mesh=geometry,vertex_colors=np.asarray(attributes['color']).copy())
        elif not hasattr(geometry.visual,'vertex_colors'):
            raise ValueError('Viewer export requires actual vertex colors; textured meshes need a texture-preserving path')
    return scene.to_geometry()


def metric_scale(a, b, true_meters):
    distance = float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))
    if not np.isfinite(distance) or distance <= 1e-8 or not np.isfinite(true_meters) or true_meters <= 0:
        raise ValueError('Calibration requires two distinct finite points and a positive measured distance')
    scale=float(true_meters / distance)
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('Calibration scale must remain finite and positive')
    return scale


def camera_records(frames, transform):
    T = np.asarray(transform, float)
    if T.shape != (4,4) or not np.isfinite(T).all() or not np.allclose(T[3], [0,0,0,1]):
        raise ValueError('Invalid viewer transform')
    if not np.allclose(T[:3,:3].T @ T[:3,:3], np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(T[:3,:3]),1):
        raise ValueError('Viewer transform must preserve relative scale and handedness')
    cameras, path = [], []
    for i, f in enumerate(sorted(frames, key=lambda x:x['timestamp_seconds'])):
        pose = T @ np.asarray(f['c2w'], float) @ np.diag([1,-1,-1,1])
        if pose.shape != (4,4) or not np.isfinite(pose).all():
            raise ValueError('Invalid camera pose')
        time = float(f['timestamp_seconds'])
        record=dict(frame_id=f['frame_id'], time=time, matrix=pose.tolist(), position=pose[:3,3].tolist())
        if 'K' in f and 'image_size' in f:
            fy=float(f['K'][1][1]);height=float(f['image_size'][1])
            if fy<=0 or height<=0 or not np.isfinite([fy,height]).all():raise ValueError('Invalid camera intrinsics')
            record['vertical_fov_degrees']=float(np.degrees(2*np.arctan(height/(2*fy))))
        cameras.append(record)
        path.append(dict(point=pose[[0,2],3].tolist(), time=time, index=i,
                         break_before=i==0 or time-cameras[i-1]['time']>2))
    if not cameras:
        raise ValueError('No cameras for component')
    return cameras, path


def _face_batches(vertices, faces, batch=100_000):
    for offset in range(0,len(faces),batch):
        tri = np.asarray(vertices)[np.asarray(faces)[offset:offset+batch]]
        cross = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
        length = np.linalg.norm(cross,axis=1)
        yield tri, cross, length


def estimate_floor(vertices, faces, camera_heights):
    camera_y = float(np.median(camera_heights))
    ys, areas = [], []
    span = float(np.ptp(np.asarray(vertices)[:,1]))
    for tri, cross, length in _face_batches(vertices,faces):
        y = tri[:,:,1].mean(1)
        mask = (np.abs(cross[:,1]) > .92 * length) & (length>1e-12) & (y < camera_y-.03*span)
        ys.append(y[mask]);areas.append(length[mask]/2)
    ys,areas = np.concatenate(ys),np.concatenate(areas)
    if not len(ys) or areas.sum()<=0:
        raise ValueError('No supported horizontal floor below cameras; manual floor height required')
    width = max(span/100,1e-5)
    hist, edges = np.histogram(ys,bins=np.arange(ys.min()-width,ys.max()+2*width,width),weights=areas)
    peak = int(hist.argmax());mask = (ys>=edges[peak]) & (ys<edges[peak+1])
    return float(np.average(ys[mask],weights=areas[mask]))


def section_segments(vertices, faces, height):
    """Intersect near-vertical triangles with Y=height, return X/Z segments."""
    if not np.isfinite(height):raise ValueError('Finite section height required')
    result=[]
    for tri,cross,length in _face_batches(vertices,faces):
        y=tri[:,:,1]
        keep=(y.min(1)<height) & (y.max(1)>height) & (length>1e-12) & (np.abs(cross[:,1])<.65*length)
        tri=tri[keep]
        if not len(tri):continue
        starts=tri[:,[0,1,2]];ends=tri[:,[1,2,0]]
        dy=ends[:,:,1]-starts[:,:,1]
        t=np.divide(height-starts[:,:,1],dy,out=np.full_like(dy,-1.),where=np.abs(dy)>1e-12)
        valid=(t>=0)&(t<1)&(np.abs(dy)>1e-12)
        usable=valid.sum(1)==2
        points=(starts+t[:,:,None]*(ends-starts))[usable]
        points=points[valid[usable]].reshape(-1,2,3)[:,:,[0,2]]
        result.append(points[np.linalg.norm(points[:,1]-points[:,0],axis=1)>1e-8])
    return np.concatenate(result) if result else np.empty((0,2,2),float)
