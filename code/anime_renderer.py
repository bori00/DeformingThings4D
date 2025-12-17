import bpy
import bmesh
import os
# --- CHANGE START ---
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
# --- CHANGE END ---
import numpy as np
import mathutils
import cv2
import sys
import time
from mathutils import Matrix, Vector, Quaternion, Euler
from mathutils.bvhtree import BVHTree

D = bpy.data
C = bpy.context
pi = 3.14


def write_flo(filename, flow):
    """
    Writes a .flo file (Middlebury format) compatible with RAFT/MS-RAFT-3D.
    flow: numpy array of shape (height, width, 2)
    """
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = flow.shape[0:2]
    magic.tofile(f)
    np.array([width], dtype=np.int32).tofile(f)
    np.array([height], dtype=np.int32).tofile(f)
    flow.astype(np.float32).tofile(f)
    f.close()


def opencv_to_blender(T):
    origin = np.array(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))
    return np.matmul(T, origin)


def blender_to_opencv(T):
    transform = np.array(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))
    return np.matmul(T, transform)


def set_camera(bpy_cam, angle=pi / 3, W=600, H=500):
    bpy_cam.angle = angle
    bpy_scene = bpy.context.scene
    bpy_scene.render.resolution_x = W
    bpy_scene.render.resolution_y = H

def look_at(obj_camera, point):
    loc_camera = obj_camera.matrix_world.to_translation()
    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()

def get_calibration_matrix_K_from_blender(camd):
    '''
    refer to: https://blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
    the code from the above link is wrong, it cause a slight error for fy in 'HORIZONTAL' mode or fx in "VERTICAL" mode.
    We did change to fix this.
    '''
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
        s_u = s_v / pixel_aspect_ratio
    else:
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = s_u / pixel_aspect_ratio
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0
    K = Matrix(((alpha_u, skew, u_0), (0, alpha_v, v_0), (0, 0, 1)))
    return K


def anime_read(filename):
    """
    filename: path of .anime file
    return:
        nf: number of frames in the animation
        nv: number of vertices in the mesh (mesh topology fixed through frames)
        nt: number of triangle face in the mesh
        vert_data: vertice data of the 1st frame (3D positions in x-y-z-order)
        face_data: riangle face data of the 1st frame
        offset_data: 3D offset data from the 2nd to the last frame
    """
    f = open(filename, 'rb')
    nf = np.fromfile(f, dtype=np.int32, count=1)[0]
    nv = np.fromfile(f, dtype=np.int32, count=1)[0]
    nt = np.fromfile(f, dtype=np.int32, count=1)[0]
    vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
    face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
    offset_data = np.fromfile(f, dtype=np.float32, count=-1)
    '''check data consistency'''
    if len(offset_data) != (nf - 1) * nv * 3:
        raise ("data inconsistent error!", filename)
    vert_data = vert_data.reshape((-1, 3))
    face_data = face_data.reshape((-1, 3))
    offset_data = offset_data.reshape((nf - 1, nv, 3))
    return nf, nv, nt, vert_data, face_data, offset_data

class AnimeRenderer:

    def __init__(self, anime_file, dum_path):
        _, _, _, vert_data, face_data, offset_data = anime_read(anime_file)
        offset_data = np.concatenate([np.zeros((1, offset_data.shape[1], offset_data.shape[2])), offset_data], axis=0)
        vertices = vert_data.tolist()
        edges = []
        faces = face_data.tolist()
        mesh_data = bpy.data.meshes.new('mesh_data')
        mesh_data.from_pydata(vertices, edges, faces)
        mesh_data.update()
        the_mesh = bpy.data.objects.new('the_mesh', mesh_data)
        bpy.context.collection.objects.link(the_mesh)
        self.the_mesh = the_mesh
        self.offset_data = offset_data
        self.vert_data = vert_data
        self.dum_path = dum_path

    def depthflowgen(self, flow_skip=1, render_sflow=True):
        num_frame = self.offset_data.shape[0]
        camera = D.objects["Camera"]

        # Create directories
        for sub in ["depth", "sflow", "mask", "optical_flow", "rgb"]:
            path = os.path.join(self.dum_path, sub)
            if not os.path.exists(path): os.makedirs(path)

        # Camera Setup
        K = get_calibration_matrix_K_from_blender(camera.data)
        fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
        width, height = C.scene.render.resolution_x, C.scene.render.resolution_y
        cam_blender = np.array(camera.matrix_world)
        cam_opencv = blender_to_opencv(cam_blender)
        cam_opencv_inv = np.linalg.inv(cam_opencv)
        world_to_cam_rot = Matrix(cam_opencv_inv[:3, :3])

        u, v = np.meshgrid(range(width), range(height))
        u = u.reshape(-1)
        v = v.reshape(-1)
        pix_position = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1)
        cam_rotation = cam_opencv[:3, :3]
        ray_direction = np.matmul(cam_rotation, pix_position.transpose()).transpose()
        ray_direction = ray_direction / np.linalg.norm(ray_direction, axis=1, keepdims=True)
        ray_origin = cam_opencv[:3, 3:].transpose()

        np.savetxt(os.path.join(self.dum_path, "cam_intr.txt"), np.array(K))
        np.savetxt(os.path.join(self.dum_path, "cam_extr.txt"), cam_opencv)

        print(f"Starting render for {num_frame} frames...")

        for src_frame_id in range(num_frame):
            print(f"Processing Frame: {src_frame_id}")
            tgt_frame_id = src_frame_id + flow_skip
            src_offset = self.offset_data[src_frame_id]
            flow_exist = (0 <= tgt_frame_id < num_frame)

            if flow_exist:
                tgt_offset = self.offset_data[tgt_frame_id]
                vert_motion = tgt_offset - src_offset

            # Update Mesh
            bm = bmesh.new()
            bm.from_mesh(self.the_mesh.data)
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()

            for i in range(len(bm.verts)):
                bm.verts[i].co = Vector(self.vert_data[i] + src_offset[i])
            bm.to_mesh(self.the_mesh.data)
            self.the_mesh.data.update()

            # Ray Casting
            raycast_mesh = self.the_mesh
            ray_begin_local = raycast_mesh.matrix_world.inverted() @ Vector(ray_origin[0])
            depsgraph = bpy.context.evaluated_depsgraph_get()
            bvhtree = BVHTree.FromObject(raycast_mesh, depsgraph)

            pcl = np.zeros_like(ray_direction)
            sflow = np.zeros_like(ray_direction)
            optical_flow = np.zeros((ray_direction.shape[0], 2), dtype=np.float32)
            valid_mask = np.zeros((ray_direction.shape[0]), dtype=np.uint8)
            rgb_buffer = np.zeros((ray_direction.shape[0], 3), dtype=np.uint8)  # BGR for OpenCV

            for i in range(ray_direction.shape[0]):
                position, norm, faceID, _ = bvhtree.ray_cast(ray_begin_local, Vector(ray_direction[i]), 50)

                if position:
                    valid_mask[i] = 255

                    # 1. Depth (Point Cloud)
                    p_world = raycast_mesh.matrix_world @ position
                    p_cam = Matrix(cam_opencv).inverted() @ p_world
                    pcl[i] = p_cam

                    # 2. RGB Shading (Normal Map)
                    norm_world = raycast_mesh.matrix_world.to_3x3() @ norm
                    norm_cam = world_to_cam_rot @ norm_world
                    norm_cam.normalize()
                    # Map [-1, 1] -> [0, 255]
                    r = int((norm_cam.x * 0.5 + 0.5) * 255)
                    g = int((norm_cam.y * 0.5 + 0.5) * 255)
                    b = int((norm_cam.z * 0.5 + 0.5) * 255)
                    rgb_buffer[i] = [b, g, r]

                    # 3. Flow
                    if render_sflow and flow_exist:
                        face = bm.faces[faceID]
                        vert_index = [v.index for v in face.verts]
                        vert_vector = [v.co for v in face.verts]
                        weights = np.array(mathutils.interpolate.poly_3d_calc(vert_vector, position))
                        flow_obj = (vert_motion[vert_index] * weights.reshape([3, 1])).sum(axis=0)

                        # Scene Flow (Camera Space)
                        # We calculate sflow in object space here for interpolation, but convert to camera space later
                        # Note: The original code stored 'flow_vector' (Object Space) in 'sflow' array
                        # and rotated it at the very end. We keep that logic to match.
                        sflow[i] = flow_obj

                        # Optical Flow
                        flow_world = raycast_mesh.matrix_world.to_3x3() @ Vector(flow_obj)
                        flow_cam = world_to_cam_rot @ flow_world

                        p_next_cam = p_cam + flow_cam

                        # Project P_t
                        if p_cam.z != 0:
                            u_curr = fx * (p_cam.x / p_cam.z) + cx
                            v_curr = fy * (p_cam.y / p_cam.z) + cy
                        else:
                            u_curr, v_curr = 0, 0

                        # Project P_t+1
                        if p_next_cam.z != 0:
                            u_next = fx * (p_next_cam.x / p_next_cam.z) + cx
                            v_next = fy * (p_next_cam.y / p_next_cam.z) + cy
                        else:
                            u_next, v_next = 0, 0

                        optical_flow[i] = [u_next - u_curr, v_next - v_curr]

            bm.free()

            # Save Outputs
            h_im, w_im = height, width

            # Mask
            cv2.imwrite(os.path.join(self.dum_path, "mask", f"{src_frame_id:04d}.png"),
                        valid_mask.reshape((h_im, w_im)))

            # RGB
            cv2.imwrite(os.path.join(self.dum_path, "rgb", f"{src_frame_id:04d}.jpg"),
                        rgb_buffer.reshape((h_im, w_im, 3)))

            # Depth
            depth = (pcl[:, 2].reshape((h_im, w_im)) * 1000).astype(np.uint16)
            cv2.imwrite(os.path.join(self.dum_path, "depth", f"{src_frame_id:04d}.png"), depth)

            if render_sflow and flow_exist:
                # --- CHANGE START ---
                # Prepare Optical Flow (H, W, 2)
                opt_flow_reshaped = optical_flow.reshape((h_im, w_im, 2))

                # Save .flo
                write_flo(os.path.join(self.dum_path, "optical_flow", f"{src_frame_id:04d}_{tgt_frame_id:04d}.flo"),
                          opt_flow_reshaped)

                # Save .npz (Optical Flow)
                np.savez_compressed(
                    os.path.join(self.dum_path, "optical_flow", f"{src_frame_id:04d}_{tgt_frame_id:04d}.npz"),
                    flow=opt_flow_reshaped)

                # Rotate Scene Flow to Camera Space
                sflow = np.matmul(np.linalg.inv(cam_opencv[:3, :3]), sflow.transpose()).transpose()
                sflow_reshaped = sflow.reshape((h_im, w_im, 3)).astype(np.float32)

                # Save .exr (Scene Flow)
                cv2.imwrite(os.path.join(self.dum_path, "sflow", f"{src_frame_id:04d}_{tgt_frame_id:04d}.exr"),
                            sflow_reshaped)

                # Save .npz (Scene Flow)
                np.savez_compressed(os.path.join(self.dum_path, "sflow", f"{src_frame_id:04d}_{tgt_frame_id:04d}.npz"),
                                    flow=sflow_reshaped)
                # --- CHANGE END ---


if __name__ == '__main__':
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]
    anime_file = argv[0]
    dump_path = argv[1]
    flow_skip = int(argv[2])

    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Cube'].select_set(state=True)
    bpy.ops.object.delete(use_global=False)

    #####################################################################
    """simply setup the camera"""
    H = 500
    W = 600
    bpy_camera = D.objects['Camera']
    bpy_camera.location, look_at_point = Vector ((2,-2,2)), Vector((0,0,1)) # need to compute this for optimal view point
    look_at(bpy_camera, look_at_point)
    set_camera(bpy_camera.data, angle=pi /3, W=W, H=H)
    bpy.context.view_layer.update() #update camera params

    renderer = AnimeRenderer(anime_file, dump_path)
    renderer.depthflowgen(flow_skip=flow_skip)