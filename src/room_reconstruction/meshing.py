"""CPU TSDF surface integration in relative scene units; no metric conversion."""
import numpy as np
from .camera_geometry import validate_pose


def integration_arrays(frame, mask, depth_trunc):
    depth=np.asarray(frame['depth']);rgb=np.asarray(frame['processed_rgb'])
    K=np.asarray(frame['K'],float);E=np.asarray(frame['w2c'],float);mask=np.asarray(mask,bool)
    if E.shape==(3,4):E=np.vstack([E,[0,0,0,1]])
    validate_pose(E)
    if (depth.ndim!=2 or mask.shape!=depth.shape or rgb.shape!=(*depth.shape,3) or rgb.dtype!=np.uint8
            or K.shape!=(3,3) or not np.isfinite(K).all() or min(K[0,0],K[1,1])<=0
            or not np.allclose(K[2],[0,0,1]) or not np.isfinite(depth_trunc) or depth_trunc<=0):
        raise ValueError('Invalid RGB/depth/mask/camera integration contract')
    valid=mask & np.isfinite(depth) & (depth>0) & (depth<depth_trunc)
    return np.ascontiguousarray(np.where(valid,depth,0),dtype=np.float32),np.ascontiguousarray(rgb),K,E


def integrate_mesh(frames, voxel_length, sdf_trunc, depth_trunc):
    if not np.isfinite([voxel_length,sdf_trunc]).all() or voxel_length<=0 or sdf_trunc<=voxel_length:
        raise ValueError('TSDF truncation must exceed positive voxel length')
    import open3d as o
    volume=o.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel_length,sdf_trunc=sdf_trunc,
        color_type=o.pipelines.integration.TSDFVolumeColorType.RGB8,depth_sampling_stride=2)
    for frame,mask in frames:
        depth,rgb,K,E=integration_arrays(frame,mask,depth_trunc)
        h,w=depth.shape
        rgbd=o.geometry.RGBDImage.create_from_color_and_depth(o.geometry.Image(rgb),o.geometry.Image(depth),
            depth_scale=1.,depth_trunc=depth_trunc,convert_rgb_to_intensity=False)
        intrinsic=o.camera.PinholeCameraIntrinsic(w,h,K[0,0],K[1,1],K[0,2],K[1,2])
        volume.integrate(rgbd,intrinsic,E)
    mesh=volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles();mesh.remove_duplicated_triangles();mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh
