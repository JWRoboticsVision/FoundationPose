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

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from estimater import *


PROJ_ROOT = Path(__file__).parent.resolve()

RS_CAMERAS = [
    "105322251564",
    "043422252387",
    "037522251142",
    "105322251225",
    "108222250342",
    "117222250549",
    "046122250168",
    "115422250549",
]

VALID_SERIALS = [
    RS_CAMERAS[i]
    for i in [
        0,
        1,
        3,
        4,
        5,
        6,
    ]
]

cvcam_in_glcam = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


class MyDataReader:
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
            # str(self._models_folder / f"{object_id}/cleaned_mesh_2000.obj")
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
        file = self._data_folder / f"{serial}/color_{frame_id:06d}.jpg"
        color = cv2.imread(str(file))
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        return color

    def get_depth(self, serial, frame_id):
        """Get depth image in numpy format, dtype=float32, [H, W]"""
        file = self._data_folder / f"{serial}/depth_{frame_id:06d}.png"
        depth = cv2.imread(str(file), -1) / 1000.0
        depth[(depth < 0.1) | (depth >= self._zfar)] = 0
        return depth

    def get_mask(self, serial, frame_id, object_idx, kernel_size=0):
        """Get mask image in numpy format, dtype=uint8, [H, W]"""
        mask_file = (
            self._data_folder
            / f"processed/segmentation/sam2/{serial}/mask"
            / f"mask_{frame_id:06d}.png"
        )
        if not mask_file.exists():
            return np.zeros((self._rs_height, self._rs_width), dtype=np.uint8)
        mask = cv2.imread(str(mask_file), cv2.IMREAD_ANYDEPTH)
        mask = (mask == (object_idx + 1)).astype(np.uint8)
        if kernel_size > 0:
            mask = self._erode_mask(mask, kernel_size)

        return mask

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


def get_bbox_from_mask(mask, margin=0):
    # Find non-zero mask pixels
    y_indices, x_indices = np.nonzero(mask)

    # Calculate the bounding box
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    # Apply margin
    x_min = max(x_min - margin, 0)
    x_max = min(x_max + margin, mask.shape[1] - 1)
    y_min = max(y_min - margin, 0)
    y_max = min(y_max + margin, mask.shape[0] - 1)

    return x_min, x_max, y_min, y_max


def get_cropped_rgb_from_color_mask(color, mask, margin=0):
    if np.sum(mask) < 100:
        return color.copy()

    y_indices, x_indices = np.nonzero(mask)
    # Calculate the bounding box
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # # Calculate the extend
    # extend = max(x_max - x_min, y_max - y_min)
    # # Apply margin
    # x_min = max(x_min - margin, 0)
    # x_max = min(x_min + extend + margin, mask.shape[1] - 1)
    # y_min = max(y_min - margin, 0)
    # y_max = min(y_min + extend + margin, mask.shape[0] - 1)

    # Calculate the center of the bounding box
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    # Calculate the maximum extend (to make it square and centered)
    half_extend = (max(x_max - x_min, y_max - y_min) + margin) // 2
    # Calculate new bounding box centered around the object
    x_min = max(x_center - half_extend, 0)
    x_max = min(x_center + half_extend, mask.shape[1] - 1)
    y_min = max(y_center - half_extend, 0)
    y_max = min(y_center + half_extend, mask.shape[0] - 1)

    # Get the cropped color
    crop = np.zeros_like(color)
    crop[y_min : y_max + 1, x_min : x_max + 1] = color[
        y_min : y_max + 1, x_min : x_max + 1
    ].copy()
    return crop


def write_pose_to_file(save_path, pose):
    np.savetxt(str(save_path), pose, fmt="%.8f")


def runner_pose_estimation(
    reader,
    estimator,
    object_idx,
    save_folder,
    est_refine_iter,
    track_refine_iter,
    mask_ratio=0.5,
    start_frame=0,
    end_frame=None,
):
    rs_width = reader.rs_width
    rs_height = reader.rs_height
    num_frames = reader.num_frames
    if end_frame is None:
        end_frame = num_frames
    object_id = reader.object_ids[object_idx]
    object_textured_mesh = trimesh.load(reader.object_files[object_idx], process=False)
    object_cleaned_mesh = trimesh.load(
        reader.object_clean_files[object_idx], process=False
    )

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

    for serial_idx, serial in enumerate(VALID_SERIALS):
        K = reader.Ks[reader.rs_serials.index(serial)]

        scene.clear()
        camera_node = scene.add(
            pyrender.IntrinsicsCamera(
                fx=K[0, 0],
                fy=K[1, 1],
                cx=K[0, 2],
                cy=K[1, 2],
                znear=0.01,
                zfar=10.0,
            ),
            pose=np.eye(4),
        )
        mesh_node = scene.add(
            pyrender.Mesh.from_trimesh(object_textured_mesh),
            pose=np.eye(4),
            parent_node=camera_node,
        )

        # get max mask area
        masks = []
        max_area = 0
        for frame_id in range(num_frames):
            mask = reader.get_mask(serial, frame_id, object_idx, kernel_size=3)
            max_area = max(max_area, np.sum(mask))
            masks.append(mask)
        mask_area_threshold = int(max_area * mask_ratio)

        fd_poses = []
        vis_images = []
        ob_in_cam_refined = np.full((4, 4), -1, dtype=np.float32)

        for frame_id in range(start_frame, end_frame):
            color = reader.get_color(serial, frame_id)
            depth = reader.get_depth(serial, frame_id)
            mask = masks[frame_id]
            rgb = color
            valid_depth_count = ((mask > 0) & (depth > 0.1)).sum()

            if mask.sum() == 0:
                depth = np.zeros_like(depth)
                mask = None

            if frame_id == start_frame or valid_depth_count <= mask_area_threshold:
                init_ob_pos_center = reader.get_init_translation(
                    frame_id, VALID_SERIALS, object_idx
                )[0][serial_idx]

                ob_in_cam_refined = estimator.register(
                    rgb=rgb,
                    depth=depth,
                    ob_mask=mask,
                    K=K,
                    iteration=est_refine_iter,
                    init_ob_pos_center=init_ob_pos_center,
                )
            else:
                ob_in_cam_refined = estimator.track_one(
                    rgb=rgb,
                    depth=depth,
                    K=K,
                    iteration=track_refine_iter,
                    prev_pose=ob_in_cam_refined,
                )

            fd_poses.append(ob_in_cam_refined)

            # render vis images
            if np.all(ob_in_cam_refined == -1):
                rendered_rgb = np.zeros_like(color)
            else:
                ob_in_glcam = cvcam_in_glcam.dot(ob_in_cam_refined)
                scene.set_pose(mesh_node, ob_in_glcam)
                rendered_rgb, _ = renderer.render(scene)
            vis = cv2.addWeighted(color, 0.3, rendered_rgb, 0.7, 0)
            vis_images.append(vis)

        fd_poses = np.stack(fd_poses, axis=0, dtype=np.float32)

        # write poses to file
        save_pose_folder = save_folder / "ob_in_cam" / object_id / serial
        # make_clean_folder(save_pose_folder)
        save_pose_folder.mkdir(parents=True, exist_ok=True)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    write_pose_to_file,
                    save_pose_folder / f"{frame_id:06d}.txt",
                    fd_poses[frame_id - start_frame],
                ): frame_id
                for frame_id in range(start_frame, end_frame)
            }

            for future in concurrent.futures.as_completed(futures):
                future.result()

        # save vis images
        save_debug_vis_folder = save_folder / f"vis/{object_id}/{serial}"
        make_clean_folder(save_debug_vis_folder)
        # save_debug_vis_folder.mkdir(parents=True, exist_ok=True)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    write_rgb_image,
                    save_debug_vis_folder / f"{frame_id:06d}.png",
                    vis_images[frame_id - start_frame],
                ): frame_id
                for frame_id in range(start_frame, end_frame)
            }

            for future in concurrent.futures.as_completed(futures):
                future.result()

        del fd_poses

        # # save vis video
        # vis_images = [None] * num_frames
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = {
        #         executor.submit(
        #             read_rgb_image,
        #             save_debug_vis_folder / f"{frame_id:06d}.png",
        #         ): frame_id
        #         for frame_id in range(num_frames)
        #     }

        #     for future in concurrent.futures.as_completed(futures):
        #         vis_images[futures[future]] = future.result()

        # save_debug_vid_video_path = save_folder / f"vis/{object_id}/{serial}.mp4"
        # create_video_from_rgb_images(save_debug_vid_video_path, vis_images, fps=30)

        del vis_images

    renderer.delete()


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence_folder", type=str, required=True, help="sequence folder"
    )
    parser.add_argument(
        "--object_idx", type=int, required=True, help="object index [1, 2, 3, 4]"
    )
    parser.add_argument("--start_frame", type=int, default=0, help="start frame index")
    parser.add_argument("--end_frame", type=int, default=None, help="end frame index")
    parser.add_argument(
        "--running_mode",
        type=str,
        default="tracking",
        choices=["tracking", "register"],
        help="est refine iter",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = args_parser()
    sequence_folder = Path(args.sequence_folder).resolve()
    object_idx = args.object_idx
    start_frame = args.start_frame
    end_frame = args.end_frame
    running_mode = args.running_mode

    est_refine_iter = 15
    track_refine_iter = 5
    if running_mode == "register":
        mask_ratio = 1.0  # for register mode only
    else:
        mask_ratio = 0.0  # for tracking mode only
    object_idx = object_idx - 1

    debug = 1
    debug_dir = PROJ_ROOT / "debug"
    make_clean_folder(debug_dir)
    make_clean_folder(debug_dir / "track_vis")
    make_clean_folder(debug_dir / "ob_in_cam")

    t_start = time.time()

    reader = MyDataReader(sequence_folder)
    save_folder = sequence_folder / "processed" / "foundationpose"
    save_folder.mkdir(parents=True, exist_ok=True)

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
        save_folder=save_folder,
        est_refine_iter=est_refine_iter,
        track_refine_iter=track_refine_iter,
        mask_ratio=mask_ratio,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    logging.info(f"done!!! time: {time.time() - t_start:.3f}s,")
