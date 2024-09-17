# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"  # GPU-based offscreen rendering

import open3d as o3d
import trimesh
import pyrender
from pathlib import Path
import argparse
import json
import shutil
import matplotlib.pyplot as plt

import concurrent.futures
import multiprocessing
import av
from scipy.spatial.transform import Rotation as R

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from estimater import *


PROJ_ROOT = Path(__file__).parent.resolve()

cvcam_in_glcam = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


class HOCapRenderer:
    def __init__(self) -> None:
        self._scene = pyrender.Scene(
            bg_color=[0, 0, 0, 1],
            ambient_light=[1.0, 1.0, 1.0, 1.0],
        )

        self._world_node = self._scene.add_node(pyrender.Node(name="world"))
        self._nodes = {}

    def add_camera(self, cam_K, cam_RT, name):
        camera = pyrender.IntrinsicsCamera(
            fx=cam_K[0, 0],
            fy=cam_K[1, 1],
            cx=cam_K[0, 2],
            cy=cam_K[1, 2],
            znear=0.01,
            zfar=10.0,
        )
        camera_node = self._scene.add(camera, pose=cam_RT @ cvcam_in_glcam, name=name)
        self._nodes[name] = camera_node

    def add_object(self, mesh, name):
        object_node = self._scene.add(
            pyrender.Mesh.from_trimesh(mesh),
            parent_node=self._world_node,
            name=name,
            pose=np.eye(4),
        )
        self._nodes[name] = object_node

    def set_node_pose(self, name, pose):
        self._scene.set_pose(self._nodes[name], pose)

    def get_rendered_image(self, width, height, cam_name, get_depth=False):
        r = pyrender.OffscreenRenderer(width, height)
        self._scene.main_camera_node = self._nodes[cam_name]
        color, depth = r.render(self._scene)
        r.delete()

        if get_depth:
            return color, depth
        else:
            return color


class HOCapReader:
    def __init__(self, sequence_folder, zfar=np.inf) -> None:
        self._data_folder = Path(sequence_folder).resolve()
        self._calib_folder = self._data_folder.parents[1] / "calibration"
        self._models_folder = self._data_folder.parents[1] / "models"
        self._zfar = zfar

        # load metadata
        self._load_metadata()

        # load intrinsics
        self._load_intrinsics()

        # load extrinsics
        self._load_extrinsics()

    def _depth2xyz(self, depth, K, T):
        """Convert depth image to xyz point cloud in camera coordinate system

        Args:
            depth (np.ndarray): depth image, dtype=float32, [H, W]
            K (np.ndarray): camera intrinsics, [3, 3]
            T (np.ndarray): camera extrinsics, [4, 4]

        Returns:
            np.ndarray: point cloud, dtype=float32, [N, 3]
        """
        H, W = depth.shape[:2]
        vs, us = np.meshgrid(
            np.arange(0, H), np.arange(0, W), sparse=False, indexing="ij"
        )
        vs = vs.reshape(-1)
        us = us.reshape(-1)
        zs = depth[vs, us]
        xs = (us - K[0, 2]) * zs / K[0, 0]
        ys = (vs - K[1, 2]) * zs / K[1, 1]
        pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N,3)
        pts = pts @ T[:3, :3].T + T[:3, 3]
        return pts

    def get_init_translation(self, frame_id, serials, object_idx):
        masks = [
            self.get_mask(serial, frame_id, object_idx, kernel_size=3)
            for serial in self._rs_serials
        ]
        depths = [self.get_depth(serial, frame_id) for serial in self._rs_serials]

        pts = [
            self._depth2xyz(depth, K, extr)
            for depth, K, extr in zip(depths, self._Ks, self._extr2world)
        ]
        pts = [pt[mask.flatten().astype(bool)] for pt, mask in zip(pts, masks)]
        pts = np.concatenate(pts, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        # remove outliers
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=300, std_ratio=2.0)
        pcd = pcd.select_by_index(ind)

        if np.asarray(pcd.points).shape[0] < 100:
            # return np.full((len(serials), 3), -1, dtype=np.float32)
            return [None] * len(serials), pcd

        center = pcd.get_center()
        # transform to each camera coordinate system
        centers = []
        for serial in serials:
            extr = self._extr2world_inv[self._rs_serials.index(serial)]
            center_cam = center @ extr[:3, :3].T + extr[:3, 3]
            centers.append(center_cam)
        centers = np.stack(centers, axis=0)
        return centers, pcd

    def _read_data_from_json(self, file):
        with open(file, "r") as f:
            data = json.load(f)
        return data

    def _erode_mask(self, mask, kernel_size=3):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        return mask

    def _load_metadata(self):
        data = self._read_data_from_json(self._data_folder / "meta.json")
        # sequnece
        self._num_frames = data["num_frames"]
        self._object_ids = data["object_ids"]
        # realsense
        self._rs_serials = data["realsense"]["serials"]
        self._rs_width = data["realsense"]["width"]
        self._rs_height = data["realsense"]["height"]
        self._num_cameras = len(self._rs_serials)
        # textured model files
        self._object_files = [
            str(self._models_folder / f"{object_id}/textured_mesh.obj")
            for object_id in self._object_ids
        ]
        self._object_clean_files = [
            str(self._models_folder / f"{object_id}/cleaned_mesh_10000.obj")
            for object_id in self._object_ids
        ]
        # extrinsic files
        self._extr_file = (
            self._calib_folder
            / "extrinsics"
            / data["calibration"]["extrinsics"]
            / "extrinsics.json"
        )

    def _load_intrinsics(self):
        def read_K_from_json(serial):
            json_path = (
                self._calib_folder
                / "intrinsics"
                / f"{serial}_{self._rs_width}x{self._rs_height}.json"
            )
            data = self._read_data_from_json(json_path)
            K = np.array(
                [
                    [data["color"]["fx"], 0.0, data["color"]["ppx"]],
                    [0.0, data["color"]["fy"], data["color"]["ppy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            return K

        self._Ks = np.stack(
            [read_K_from_json(serial) for serial in self._rs_serials], axis=0
        )

    def _load_extrinsics(self):
        with open(self._extr_file, "r") as f:
            data = json.load(f)
        tag_1 = np.array(
            [
                [
                    data["extrinsics"]["tag_1"][0],
                    data["extrinsics"]["tag_1"][1],
                    data["extrinsics"]["tag_1"][2],
                    data["extrinsics"]["tag_1"][3],
                ],
                [
                    data["extrinsics"]["tag_1"][4],
                    data["extrinsics"]["tag_1"][5],
                    data["extrinsics"]["tag_1"][6],
                    data["extrinsics"]["tag_1"][7],
                ],
                [
                    data["extrinsics"]["tag_1"][8],
                    data["extrinsics"]["tag_1"][9],
                    data["extrinsics"]["tag_1"][10],
                    data["extrinsics"]["tag_1"][11],
                ],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        tag_1_inv = np.linalg.inv(tag_1)

        self._extr2master = np.stack(
            [
                np.array(
                    [
                        [
                            data["extrinsics"][s][0],
                            data["extrinsics"][s][1],
                            data["extrinsics"][s][2],
                            data["extrinsics"][s][3],
                        ],
                        [
                            data["extrinsics"][s][4],
                            data["extrinsics"][s][5],
                            data["extrinsics"][s][6],
                            data["extrinsics"][s][7],
                        ],
                        [
                            data["extrinsics"][s][8],
                            data["extrinsics"][s][9],
                            data["extrinsics"][s][10],
                            data["extrinsics"][s][11],
                        ],
                        [0, 0, 0, 1],
                    ],
                    dtype=np.float32,
                )
                for s in self._rs_serials
            ],
            axis=0,
        )
        self._extr2master_inv = np.stack(
            [np.linalg.inv(t) for t in self._extr2master], axis=0
        )
        self._extr2world = np.stack([tag_1_inv @ t for t in self._extr2master], axis=0)
        self._extr2world_inv = np.stack(
            [np.linalg.inv(t) for t in self._extr2world], axis=0
        )

    def get_color(self, serial, frame_id):
        """Get RGB image in numpy format, dtype=uint8, [H, W, 3]"""
        file_path = self._data_folder / serial / f"color_{frame_id:06d}.jpg"
        color = cv2.cvtColor(cv2.imread(str(file_path)), cv2.COLOR_BGR2RGB)
        return color

    def get_depth(self, serial, frame_id):
        """Get depth image in numpy format, dtype=float32, [H, W]"""
        file_path = self._data_folder / serial / f"depth_{frame_id:06d}.png"
        depth = cv2.imread(str(file_path), -1) / 1000.0
        depth[(depth < 0.1) | (depth >= self._zfar)] = 0
        return depth

    def get_mask(self, serial, frame_id, object_idx, kernel_size=0):
        """Get mask image in numpy format, dtype=uint8, [H, W]"""
        file_path = (
            self._data_folder
            / f"processed/segmentation/sam2/{serial}/mask/mask_{frame_id:06d}.png"
        )
        if not file_path.exists():
            return np.zeros((self._rs_height, self._rs_width), dtype=np.uint8)
        mask = cv2.imread(str(file_path), cv2.IMREAD_ANYDEPTH)
        mask = (mask == (object_idx + 1)).astype(np.uint8)
        if kernel_size > 0:
            mask = self._erode_mask(mask, kernel_size)
        return mask

    def get_valid_serials(self):
        valid_serials = []

        for serial in self._rs_serials:
            if (
                self._data_folder
                / f"processed/segmentation/sam2/{serial}/mask/mask_000000.png"
            ).exists():
                valid_serials.append(serial)

        return valid_serials

    @property
    def num_frames(self):
        return self._num_frames

    @property
    def rs_serials(self):
        return self._rs_serials

    @property
    def rs_width(self):
        return self._rs_width

    @property
    def rs_height(self):
        return self._rs_height

    @property
    def object_ids(self):
        return self._object_ids

    @property
    def object_files(self):
        return self._object_files

    @property
    def object_clean_files(self):
        return self._object_clean_files

    @property
    def Ks(self):
        return self._Ks

    @property
    def extr2world(self):
        return self._extr2world

    @property
    def extr2world_inv(self):
        return self._extr2world_inv


def read_rgb_image(image_path):
    return cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)


def write_rgb_image(save_path, rgb):
    cv2.imwrite(str(save_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def create_video_from_rgb_images(video_path, rgb_images, fps=30):
    height, width = rgb_images[0].shape[:2]

    container = av.open(str(video_path), mode="w")

    stream = container.add_stream("h264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    for image in rgb_images:
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame = frame.reformat(format="yuv420p")

        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()


def make_clean_folder(folder):
    print(f"cleaning folder: {folder}")
    if Path(folder).exists():
        shutil.rmtree(str(folder))
    Path(folder).mkdir(parents=True, exist_ok=False)


def write_pose_to_file(save_path, pose):
    np.savetxt(str(save_path), pose, fmt="%.8f")


def mat_to_quat(mat):
    dtype = mat.dtype
    if np.all(mat == -1):
        return np.full(7, -1, dtype=dtype)
    r = R.from_matrix(mat[:3, :3])
    t = mat[:3, 3]
    return np.concatenate((r.as_quat(), t), axis=0).astype(dtype)


def quat_to_mat(quat):
    dtype = quat.dtype
    if np.all(quat == -1):
        return np.full((4, 4), -1, dtype=dtype)
    r = R.from_quat(quat[:4])
    t = quat[4:]
    mat = np.eye(4)
    mat[:3, :3] = r.as_matrix()
    mat[:3, 3] = t
    return mat.astype(dtype)


# TODO: implement this function
def get_best_ob_in_world(ob_in_cam_poses, cam_RTs, thresh_rot=0.1, thresh_trans=0.02):
    def is_valid_pose(ob_in_world):
        return (
            -0.6 < ob_in_world[0, 3] < 0.6
            and -0.4 < ob_in_world[1, 3] < 0.4
            and -0.015 < ob_in_world[2, 3] < 0.6
        )

    def quat_distance(q1, q2):
        """Calculate the distance between two quaternions."""
        rotation1 = q1[:4] / np.linalg.norm(q1[:4])
        rotation2 = q2[:4] / np.linalg.norm(q2[:4])
        r1 = R.from_quat(rotation1)
        r2 = R.from_quat(rotation2)
        relative_rotation = r1.inv() * r2
        return relative_rotation.magnitude()

    def trans_distance(t1, t2):
        return np.linalg.norm(t1 - t2)

    def average_quats(quats):
        """Calculate the average quaternion for a list of quaternions."""
        rotations = R.from_quat(quats)
        average_rotation = rotations.mean()
        return average_rotation.as_quat()

    valid_quat_poses = []
    for ob_in_cam, cam_RT in zip(ob_in_cam_poses, cam_RTs):
        if np.all(ob_in_cam == -1):
            continue
        ob_in_world = cam_RT @ ob_in_cam
        if is_valid_pose(ob_in_world):
            ob_in_world_quat = mat_to_quat(ob_in_world)
            valid_quat_poses.append(ob_in_world_quat)

    if len(valid_quat_poses) == 0:
        return np.full((4, 4), -1, dtype=np.float32)

    best_quat = None
    best_trans = None
    quat_dist_loss = np.inf
    trans_dist_loss = np.inf
    best_quat_count = 0
    best_trans_count = 0

    for quat_pose in valid_quat_poses:
        curr_quat = quat_pose[:4]
        curr_trans = quat_pose[4:]
        quat_dists = np.array(
            [quat_distance(curr_quat, p[:4]) for p in valid_quat_poses]
        )
        trans_dists = np.array(
            [trans_distance(curr_trans, p[4:]) for p in valid_quat_poses]
        )

        valid_quat_mask = quat_dists < thresh_rot
        valid_trans_mask = trans_dists < thresh_trans

        valid_quat_count = valid_quat_mask.sum()
        valid_trans_count = valid_trans_mask.sum()

        ################## Use the average of valid quaternions and translations ##################
        if valid_quat_count > 0:
            # Use the average of valid quaternions
            valid_quats = [
                valid_quat_poses[i][:4]
                for i in range(len(valid_quat_mask))
                if valid_quat_mask[i]
            ]
            curr_quat = average_quats(valid_quats)

        if valid_trans_count > 0:
            # Use the average of valid translations
            valid_trans = [
                valid_quat_poses[i][4:]
                for i in range(len(valid_trans_mask))
                if valid_trans_mask[i]
            ]
            curr_trans = np.mean(valid_trans, axis=0)
        ################## Use the average of valid quaternions and translations ##################

        if valid_quat_count > best_quat_count:
            best_quat = curr_quat
            best_quat_count = valid_quat_count
            quat_dist_loss = quat_dists[valid_quat_mask].mean()
        elif valid_quat_count == best_quat_count:
            if quat_dists[valid_quat_mask].mean() < quat_dist_loss:
                best_quat = curr_quat
                quat_dist_loss = quat_dists[valid_quat_mask].mean()
            elif quat_dists[valid_quat_mask].mean() == quat_dist_loss:
                best_quat = average_quats([best_quat, curr_quat])

        if valid_trans_count > best_trans_count:
            best_trans = curr_trans
            best_trans_count = valid_trans_count
            trans_dist_loss = trans_dists[valid_trans_mask].mean()
        elif valid_trans_count == best_trans_count:
            if trans_dists[valid_trans_mask].mean() < trans_dist_loss:
                best_trans = curr_trans
            elif trans_dists[valid_trans_mask].mean() == trans_dist_loss:
                best_trans = np.mean([best_trans, curr_trans], axis=0)

    if best_quat is not None and best_trans is not None:
        best_quat_pose = np.concatenate((best_quat, best_trans), axis=0).astype(
            np.float32
        )
    else:
        best_quat_pose = np.full(7, -1, dtype=np.float32)

    return quat_to_mat(best_quat_pose)


def runner_pose_estimation(
    reader,
    estimator,
    object_idx,
    save_folder,
    est_refine_iter,
    track_refine_iter,
    start_frame=0,
    end_frame=None,
    thresh_rot=0.1,
    thresh_trans=0.02,
):
    rs_width = reader.rs_width
    rs_height = reader.rs_height
    num_frames = reader.num_frames
    rs_serials = reader.rs_serials
    object_id = reader.object_ids[object_idx]
    object_textured_file = reader.object_files[object_idx]
    object_cleaned_file = reader.object_clean_files[object_idx]
    valid_serials = reader.get_valid_serials()
    valid_Ks = [reader.Ks[rs_serials.index(serial)] for serial in valid_serials]
    valid_RTs = [
        reader.extr2world[rs_serials.index(serial)] for serial in valid_serials
    ]
    valid_RTs_inv = [
        reader.extr2world_inv[rs_serials.index(serial)] for serial in valid_serials
    ]

    if end_frame is None:
        end_frame = num_frames
    else:
        end_frame = min(end_frame, num_frames)

    object_textured_mesh = trimesh.load(object_textured_file, process=False)
    object_cleaned_mesh = trimesh.load(object_cleaned_file, process=False)

    # update mesh for pose estimator
    estimator.reset_object(
        model_pts=object_cleaned_mesh.vertices,
        model_normals=object_cleaned_mesh.vertex_normals,
        mesh=object_textured_mesh,
    )

    # initialize offscreen renderer
    renderer = pyrender.OffscreenRenderer(rs_width, rs_height)
    scene = pyrender.Scene(
        bg_color=[0, 0, 0, 1],
        ambient_light=[1.0, 1.0, 1.0, 1.0],
    )
    world_node = scene.add_node(pyrender.Node(name="world"))
    camera_nodes = {
        serial: scene.add(
            pyrender.IntrinsicsCamera(
                fx=K[0, 0],
                fy=K[1, 1],
                cx=K[0, 2],
                cy=K[1, 2],
                znear=0.01,
                zfar=10.0,
            ),
            parent_node=world_node,
            name=serial,
            pose=cam_RT @ cvcam_in_glcam,
        )
        for serial, K, cam_RT in zip(reader.rs_serials, reader.Ks, reader.extr2world)
    }
    object_node = scene.add(
        pyrender.Mesh.from_trimesh(object_textured_mesh),
        parent_node=world_node,
        name=object_id,
        pose=np.eye(4),
    )

    ob_in_world_refined = np.full((4, 4), -1, dtype=np.float32)
    ob_in_cam_poses = [None for _ in range(len(valid_serials))]
    for frame_id in range(start_frame, end_frame, 1):
        init_ob_pos_centers, _ = reader.get_init_translation(
            0, valid_serials, object_idx
        )
        colors = []
        for serial_idx, serial in enumerate(valid_serials):
            color = reader.get_color(serial, frame_id)
            depth = reader.get_depth(serial, frame_id)
            mask = reader.get_mask(serial, frame_id, object_idx, kernel_size=3)
            K = valid_Ks[serial_idx]
            colors.append(color)

            if mask.sum() == 0:
                depth = np.zeros_like(depth)
                mask = None

            if frame_id == start_frame:
                ob_in_cam_mat = estimator.register(
                    rgb=color,
                    depth=depth,
                    ob_mask=mask,
                    K=K,
                    iteration=est_refine_iter,
                    init_ob_pos_center=init_ob_pos_centers[serial_idx],
                )
            else:
                prev_pose = (
                    # valid_RTs[serial_idx] @ ob_in_world_refined
                    valid_RTs_inv[serial_idx] @ ob_in_world_refined
                    if np.all(ob_in_world_refined != -1)
                    else ob_in_cam_poses[serial_idx]
                )
                ob_in_cam_mat = estimator.track_one(
                    rgb=color,
                    depth=depth,
                    K=valid_Ks[serial_idx],
                    iteration=track_refine_iter,
                    prev_pose=prev_pose,
                )

            ob_in_cam_poses[serial_idx] = ob_in_cam_mat

            # save pose to file
            save_pose_folder = save_folder / object_id / "ob_in_cam" / serial
            save_pose_folder.mkdir(parents=True, exist_ok=True)
            write_pose_to_file(
                save_pose_folder / f"{frame_id:06d}.txt", mat_to_quat(ob_in_cam_mat)
            )

            # render vis image
            if np.all(ob_in_cam_mat == -1):
                rendered_rgb = np.zeros_like(color)
            else:
                scene.main_camera_node = camera_nodes[serial]
                ob_in_world = valid_RTs[serial_idx] @ ob_in_cam_mat
                scene.set_pose(object_node, ob_in_world)
                rendered_rgb, _ = renderer.render(scene)
            vis = cv2.addWeighted(color, 0.3, rendered_rgb, 0.7, 0)

            save_vis_folder = save_folder / object_id / "vis_ob_in_cam" / serial
            save_vis_folder.mkdir(parents=True, exist_ok=True)
            write_rgb_image(save_vis_folder / f"vis_{frame_id:06d}.jpg", vis)

        # refine object pose in world coordinate system
        ob_in_world_refined = get_best_ob_in_world(
            ob_in_cam_poses,
            valid_RTs,
            thresh_rot=thresh_rot,
            thresh_trans=thresh_trans,
        )

        # save pose to file
        save_pose_folder = save_folder / object_id / "ob_in_world"
        save_pose_folder.mkdir(parents=True, exist_ok=True)
        write_pose_to_file(
            save_pose_folder / f"{frame_id:06d}.txt", mat_to_quat(ob_in_world_refined)
        )

        # render vis image
        if np.all(ob_in_world_refined == -1):
            rendered_rgbs = [np.zeros_like(color)] * len(valid_serials)
        else:
            scene.set_pose(object_node, ob_in_world_refined)
            rendered_rgbs = []
            for serial in valid_serials:
                scene.main_camera_node = camera_nodes[serial]
                rendered_rgb, _ = renderer.render(scene)
                rendered_rgbs.append(rendered_rgb)
        vis = [
            cv2.addWeighted(color, 0.3, rendered_rgb, 0.7, 0)
            for color, rendered_rgb in zip(colors, rendered_rgbs)
        ]

        vis = np.concatenate(vis, axis=1)
        save_vis_folder = save_folder / object_id / "vis_ob_in_world"
        save_vis_folder.mkdir(parents=True, exist_ok=True)
        write_rgb_image(save_vis_folder / f"vis_{frame_id:06d}.jpg", vis)

    renderer.delete()


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence_folder", type=str, required=True, help="sequence folder"
    )
    parser.add_argument(
        "--object_idx", type=int, required=True, help="object index [1, 2, 3, 4]"
    )
    parser.add_argument(
        "--est_refine_iter",
        type=int,
        default=15,
        help="number of iterations for estimation",
    )
    parser.add_argument(
        "--track_refine_iter",
        type=int,
        default=5,
        help="number of iterations for tracking",
    )
    parser.add_argument("--start_frame", type=int, default=0, help="start frame")
    parser.add_argument("--end_frame", type=int, default=None, help="end frame")
    return parser.parse_args()


if __name__ == "__main__":
    args = args_parser()
    sequence_folder = Path(args.sequence_folder).resolve()
    object_idx = args.object_idx
    est_refine_iter = args.est_refine_iter
    track_refine_iter = args.track_refine_iter
    start_frame = args.start_frame
    end_frame = args.end_frame
    est_refine_iter = args.est_refine_iter
    track_refine_iter = args.track_refine_iter

    object_idx = object_idx - 1

    debug = 1
    debug_dir = PROJ_ROOT / "debug"
    make_clean_folder(debug_dir)
    make_clean_folder(debug_dir / "track_vis")
    make_clean_folder(debug_dir / "ob_in_cam")

    t_start = time.time()

    reader = HOCapReader(sequence_folder)
    save_folder = sequence_folder / "processed" / "foundationpose_with_optim"

    set_logging_format()
    set_seed(0)

    # initialize FoundationPose estimator
    object_mesh = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4))
    estimator = FoundationPose(
        model_pts=object_mesh.vertices,
        model_normals=object_mesh.vertex_normals,
        mesh=object_mesh,
        scorer=ScorePredictor(),
        refiner=PoseRefinePredictor(),
        glctx=dr.RasterizeCudaContext(),
        debug=debug,
        debug_dir=f"{debug_dir}",
        rotation_grid_min_n_views=120,
        rotation_grid_inplane_step=15,
    )

    runner_pose_estimation(
        reader,
        estimator,
        object_idx,
        start_frame=start_frame,
        end_frame=end_frame,
        save_folder=save_folder,
        est_refine_iter=est_refine_iter,
        track_refine_iter=track_refine_iter,
        thresh_rot=0.06,  # radians, 3.4 degrees
        thresh_trans=0.015,  # meters, 1.5cm
    )

    logging.info(f"done!!! time: {time.time() - t_start:.3f}s,")
