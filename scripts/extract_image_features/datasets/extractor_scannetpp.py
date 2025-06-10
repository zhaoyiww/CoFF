import torch
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import pickle
from PIL import Image
from torchvision.transforms import transforms, InterpolationMode
from datasets.base_extractor import Processor, adjust_intrinsic
from utils.o3d_tools import get_nearest_neighbor, compute_overlap_ratio, array_transform, list2ndarray, tensor2pcd
from utils.visualization import draw_registration_result_three_views, draw_registration_result_single
import copy
import json
import shutil
import open3d as o3d


def get_depth_intrinsics(rgb_intrinsics, res_rgb, res_depth):
    """
    Get depth intrinsics if only RGB intrinsics are provided.
    """
    if res_rgb == res_depth:
        return rgb_intrinsics
    depth_intrinsics = copy.deepcopy(rgb_intrinsics)
    # Scaling factors
    sx = res_depth[0] / res_rgb[0]
    sy = res_depth[1] / res_rgb[1]

    # Scaled intrinsics for depth frame
    depth_intrinsics[0, 0] *= sx
    depth_intrinsics[1, 1] *= sy
    depth_intrinsics[0, 2] *= sx
    depth_intrinsics[1, 2] *= sy
    return depth_intrinsics


def extract_pose_cam2world(cfg, scene):
    """
    Extract camera-to-world poses and intrinsic matrices from a JSON file.
    """
    file_path = osp.join(cfg.input_root, 'image', scene, 'pose_intrinsic_imu.json')
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Pose and intrinsic file not found: {file_path}")

    with open(file_path, 'r') as file:
        data = json.load(file)

    pose_matrices = []
    intrinsic_matrices = []

    for frame, frame_data in data.items():
        pose = frame_data['pose']
        intrinsic = frame_data['intrinsic']
        # Convert to 4x4 matrix
        pose_matrix = np.array(pose)
        pose_matrices.append(pose_matrix)
        intrinsic_matrix = np.array(intrinsic)
        intrinsic_matrices.append(intrinsic_matrix)

    return pose_matrices, intrinsic_matrices



class ScanNetppDataset_Process(Processor):
    """
    Class for processing the ScanNet++ dataset.
    Handles dataset preprocessing, feature extraction, and configuration generation.
    """

    def __init__(self, config):
        # Initialize the base Processor class
        super().__init__(config)

        self.scene_list = None
        self.pkl_info = None
        self.overlap_radius = 0.025
        self.voxel_size = 0.025

        self.image_rgb_size = [1440, 1920]
        self.image_depth_size = [192, 256]

        # Generate configuration file for all pairs
        self.read_pkl()
        # Process the dataset
        self.process_dataset()

    def read_pkl(self):
        """
        Read or generate the '.pkl' configuration file for the dataset.
        """
        pkl_read_path = os.path.join(self.data_path, 'metadata', f'{self.benchmark_name}_converted.pkl')

        # Read all scenes
        scene_list_dir = os.path.join(self.data_path, 'metadata', 'split', 'test_scannetpp.txt')
        if not os.path.isfile(scene_list_dir):
            raise FileNotFoundError(f"Scene list file not found: {scene_list_dir}")

        with open(scene_list_dir, 'r') as f:
            self.scene_list = [line.strip() for line in f]

        if os.path.isfile(pkl_read_path):
            print(f"'.pkl' file already exists. Loading...")
            with open(pkl_read_path, "rb") as f:
                self.pkl_info = pickle.load(f)
        else:
            raise NotImplementedError("The '.pkl' file generation is not implemented yet.")

    def process_dataset(self):
        """
        Process the dataset by extracting image features and fusing them with point cloud data.
        """
        # Read pair information
        pkl_info = self.pkl_info
        pair_src_info = pkl_info['src'] if isinstance(pkl_info['src'], list) else pkl_info['src'].tolist()
        pair_tgt_info = pkl_info['tgt'] if isinstance(pkl_info['tgt'], list) else pkl_info['tgt'].tolist()
        pcd_info = sorted(list(set(pair_src_info + pair_tgt_info)))

        # Initialize scene and pose matrices
        self.scene = osp.basename(osp.dirname(pcd_info[0]))
        self.pose_matrices, self.intrinsic_matrices = extract_pose_cam2world(self, self.scene)

        missing_ratio_all = []

        # Loop through all point cloud files
        for pair_id, curr_pcd_dir in enumerate(tqdm(pcd_info)):
            scene_curr = osp.basename(osp.dirname(curr_pcd_dir))
            if scene_curr != self.scene:
                self.scene = scene_curr
                self.pose_matrices, self.intrinsic_matrices = extract_pose_cam2world(self, self.scene)

            # Load raw geometry point cloud
            pcd_path_input = os.path.join(self.data_path, 'geometry', curr_pcd_dir).replace('.npz', '.ply')
            if not os.path.isfile(pcd_path_input):
                print(f"Warning: Point cloud file not found: {pcd_path_input}")
                continue

            pcd_fragment = o3d.io.read_point_cloud(pcd_path_input)
            pcd_fragment = pcd_fragment.voxel_down_sample(voxel_size=self.voxel_size)
            pcd_fragment_clr = np.asarray(pcd_fragment.colors).astype(np.float32)
            pcd_fragment = torch.from_numpy(np.asarray(pcd_fragment.points)).float()

            # Create the saving path for output
            pcd_path_save = (pcd_path_input.replace(self.input_root, self.output_root)
                             .replace('geometry', 'data'))
            os.makedirs(os.path.dirname(pcd_path_save), exist_ok=True)

            # Process based on the number of images
            if self.num_images in [1, 2, 3]:
                try:
                    (pose_frame_1, pose_frame_2, pose_frame_3, color_frame_1, color_frame_2, color_frame_3,
                     depth_frame_1,
                     depth_frame_2, depth_frame_3, camera_intrinsic_1, camera_intrinsic_2, camera_intrinsic_3) = (
                        self.three_view_img_process_scannetpp(curr_pcd_dir))

                    # Adjust camera intrinsics
                    camera_intrinsic_1_adjust = adjust_intrinsic(camera_intrinsic_1, self.image_rgb_size,
                                                                 self.image_depth_resize)
                    camera_intrinsic_2_adjust = adjust_intrinsic(camera_intrinsic_2, self.image_rgb_size,
                                                                 self.image_depth_resize)
                    camera_intrinsic_3_adjust = adjust_intrinsic(camera_intrinsic_3, self.image_rgb_size,
                                                                 self.image_depth_resize)

                    # Transform images
                    color_frame_1, color_frame_2, color_frame_3 = (
                        self.three_view_img_transform(color_frame_1, color_frame_2, color_frame_3, image_type='color'))
                    depth_frame_1, depth_frame_2, depth_frame_3 = (
                        self.three_view_img_transform(depth_frame_1, depth_frame_2, depth_frame_3, image_type='depth'))

                    # Transform point cloud into frames
                    pcd_frame_1, pcd_frame_2, pcd_frame_3 = self.pcd_fragment2frame(
                        pcd_fragment, 3, pose_frame_1, pose_frame_2, pose_frame_3)

                    # Project point cloud to image and extract 2D features
                    pcd2img_frame_1, mask_1 = self.frame_pcd2image_projection(pcd_frame_1, depth_frame_1,
                                                                              camera_intrinsic_1_adjust,
                                                                              pcd_fragment_clr)
                    pcd2img_frame_2, mask_2 = self.frame_pcd2image_projection(pcd_frame_2, depth_frame_2,
                                                                              camera_intrinsic_2_adjust)
                    pcd2img_frame_3, mask_3 = self.frame_pcd2image_projection(pcd_frame_3, depth_frame_3,
                                                                              camera_intrinsic_3_adjust)

                    # Extract 2D image features
                    for param in self.backbone2d.parameters():
                        param.requires_grad = False
                    img_feats_2d_1 = self.backbone2d(color_frame_1.unsqueeze(0).to(self.device)).squeeze(0)
                    img_feats_2d_2 = self.backbone2d(color_frame_2.unsqueeze(0).to(self.device)).squeeze(0)
                    img_feats_2d_3 = self.backbone2d(color_frame_3.unsqueeze(0).to(self.device)).squeeze(0)

                    # Convert 2D features to 3D
                    img_feats_3d_1 = self.feats_idx_2d_to_3d(pcd2img_frame_1, img_feats_2d_1, mask_1)
                    img_feats_3d_2 = self.feats_idx_2d_to_3d(pcd2img_frame_2, img_feats_2d_2, mask_2)
                    img_feats_3d_3 = self.feats_idx_2d_to_3d(pcd2img_frame_3, img_feats_2d_3, mask_3)

                    # Fuse image features
                    if self.num_images == 1:
                        img_feats_final, idx_img_feats_valid, missing_ratio = self.single_view_img_feat(pcd_fragment,
                                                                                                        mask_1,
                                                                                                        img_feats_3d_1)
                    elif self.num_images == 2:
                        img_feats_final, idx_img_feats_valid, missing_ratio = (
                            self.two_view_img_feat_fusion(pcd_fragment, mask_1, mask_2, img_feats_3d_1, img_feats_3d_2))
                    elif self.num_images == 3:
                        img_feats_final, idx_img_feats_valid, missing_ratio = (
                            self.three_view_img_feat_fusion(pcd_fragment, mask_1, mask_2, mask_3, img_feats_3d_1,
                                                            img_feats_3d_2, img_feats_3d_3))

                    # Save processed data
                    np.savez_compressed(
                        pcd_path_save.replace('.ply', ''),
                        xyz=pcd_fragment,
                        rgb=pcd_fragment_clr,
                        img_feats=img_feats_final.cpu(),
                        idx_img_feats_valid=idx_img_feats_valid.cpu()
                    )
                    shutil.copy2(f'{os.path.splitext(pcd_path_input)[0]}.pose.txt', os.path.dirname(pcd_path_save))
                    missing_ratio_all.append(missing_ratio)

                except Exception as e:
                    print(f"Error processing {curr_pcd_dir}: {e}")
                    continue

        # Save missing ratios
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(pcd_path_save))), 'missing_ratio.txt'),
                  'w') as file:
            for item in missing_ratio_all:
                file.write(f"{item}\n")

    def three_view_img_process_scannetpp(self, curr_pcd_dir):
        """
        Process three-view images and their corresponding data for ScanNet++.
        """
        # Load pose information
        pose_ref = os.path.join(self.data_path, 'geometry', curr_pcd_dir.replace('npz', 'pose.txt'))
        if not os.path.isfile(pose_ref):
            raise FileNotFoundError(f"Pose file not found: {pose_ref}")
        with open(pose_ref, 'r') as f:
            info_ref = f.readlines()

        # Extract scene and pose IDs
        scene = os.path.basename(os.path.dirname(curr_pcd_dir))
        pose_id_ref_1, pose_id_ref_2 = info_ref[0].split()[2], info_ref[0].split()[3]
        pose_id_ref_3 = str((int(pose_id_ref_1) + int(pose_id_ref_2)) // 2)

        # Load camera intrinsics and poses
        camera_intrinsic_1 = self.intrinsic_matrices[int(pose_id_ref_1)]
        camera_intrinsic_2 = self.intrinsic_matrices[int(pose_id_ref_2)]
        camera_intrinsic_3 = self.intrinsic_matrices[int(pose_id_ref_3)]
        pose_frame_1 = self.pose_matrices[int(pose_id_ref_1)]
        pose_frame_2 = self.pose_matrices[int(pose_id_ref_2)]
        pose_frame_3 = self.pose_matrices[int(pose_id_ref_3)]

        # Load RGB and depth images
        filenames = [f"frame_{pose_id.zfill(6)}" for pose_id in [pose_id_ref_1, pose_id_ref_2, pose_id_ref_3]]
        scene_path = os.path.join(self.data_path, 'image', scene)
        color_frames, depth_frames = [], []

        for filename in filenames:
            # Load color image
            color_path = os.path.join(scene_path, 'rgb', f"{filename}.png")
            if not os.path.exists(color_path):
                color_path = os.path.join(scene_path, 'rgb', f"{filename}.jpg")
            if not os.path.isfile(color_path):
                raise FileNotFoundError(f"Color image not found: {color_path}")
            color_frames.append(np.array(Image.open(color_path)))

            # Load depth image
            depth_path = os.path.join(scene_path, 'depth', f"{filename}.png")
            if not os.path.exists(depth_path):
                depth_path = os.path.join(scene_path, 'depth', f"{filename}.jpg")
            if not os.path.isfile(depth_path):
                raise FileNotFoundError(f"Depth image not found: {depth_path}")
            depth_frames.append(np.array(Image.open(depth_path)))

        return (pose_frame_1, pose_frame_2, pose_frame_3, *color_frames, *depth_frames,
                camera_intrinsic_1, camera_intrinsic_2, camera_intrinsic_3)

    def pcd_fragment2frame(self, pcd_fragment, num_frame, pose_frame_1, pose_frame_2=None, pose_frame_3=None,
                           pose_frame_4=None, pose_frame_5=None, data_augmentation=False):
        """
        Transform a point cloud fragment into multiple frames based on given poses.

        Args:
            pcd_fragment (np.ndarray): Input point cloud fragment.
            num_frame (int): Number of frames (e.g., 3).
            pose_frame_1, ..., pose_frame_5 (np.ndarray): Transformation matrices for each frame.
            data_augmentation (bool): Whether to apply data augmentation (not implemented).

        Returns:
            Transformed point cloud frames as torch.FloatTensor.
        """
        pos_ref_1_rot, pose_ref_1_tra = pose_frame_1[:3, :3], pose_frame_1[:3, 3]
        pcd_frame_1 = pcd_fragment.clone().detach().float()

        if num_frame == 3:
            if pose_frame_2 is None or pose_frame_3 is None:
                raise ValueError("pose_frame_2 and pose_frame_3 must be provided for 3 frames.")

            pos_ref_2_rot, pose_ref_2_tra = pose_frame_2[:3, :3], pose_frame_2[:3, 3]
            pos_ref_3_rot, pose_ref_3_tra = pose_frame_3[:3, :3], pose_frame_3[:3, 3]

            # Transform fragment to mid-frame
            pcd_to_mid = np.matmul(pcd_fragment, pos_ref_1_rot.T) + pose_ref_1_tra

            # Transform mid-frame to frame 2 and frame 3
            pcd_frame_2 = np.matmul(pcd_to_mid - pose_ref_2_tra, pos_ref_2_rot)
            pcd_frame_3 = np.matmul(pcd_to_mid - pose_ref_3_tra, pos_ref_3_rot)

            return (
                pcd_frame_1,
                pcd_frame_2.clone().detach().float(),
                pcd_frame_3.clone().detach().float(),
            )

        raise NotImplementedError("Only 3-frame transformation is currently supported.")