import torch
import os
import numpy as np
from tqdm import tqdm
import pickle
from PIL import Image
from datasets.base_extractor import Processor, adjust_intrinsic
from utils.o3d_tools import compute_overlap_ratio, array_transform, list2ndarray
import open3d as o3d
import re


class IndoorLRSDataset_Process(Processor):
    """
    Class for processing the IndoorLRS dataset.
    Handles dataset processing, image feature extraction, and multi-view fusion.
    """

    def __init__(self, config):
        super().__init__(config)
        self.scene_list = None
        self.pkl_info = None
        self.overlap_radius = 0.1
        self.voxel_size = 0.025

        self.generate_pkl()
        self.process_dataset()

    def generate_pkl(self):
        """
        Generate a '.pkl' configuration file for the dataset.
        Reads registration log files and computes transformations, overlaps, and pair information.
        """
        pkl_read_path = os.path.join(self.data_path, 'metadata', f'{self.benchmark_name}_converted.pkl')

        # Read all scenes
        scene_list_dir = os.path.join(self.data_path, 'metadata', 'split', 'test_indoorlrs.txt')
        with open(scene_list_dir, 'r') as f:
            self.scene_list = [line.strip() for line in f.readlines()]

        if os.path.isfile(pkl_read_path):
            print(f"'.pkl' file already exists at {pkl_read_path}")
            with open(pkl_read_path, "rb") as f:
                self.pkl_info = pickle.load(f)
        else:
            pkl_save_path = os.path.join(self.save_dir, self.dataset, 'configs')
            os.makedirs(pkl_save_path, exist_ok=True)

            src, tgt, rot, trans, overlap = [], [], [], [], []

            for scene in tqdm(self.scene_list, desc="Processing scenes"):
                gt_transform_dir = os.path.join(self.data_path, 'geometry', 'test', scene, 'reg_output.log')
                if not os.path.exists(gt_transform_dir):
                    print(f"Warning: Missing registration log for scene {scene}")
                    continue

                with open(gt_transform_dir, 'r') as f:
                    info = f.readlines()

                # Extract pair IDs
                pairs_id_idx = [i for i in range(len(info)) if i % 5 == 0]
                pairs_info = np.array(info)[pairs_id_idx]

                pairs_id = np.zeros((len(pairs_info), 2), dtype=int)
                for i, pair in enumerate(pairs_info):
                    pairs_id[i, 0] = int(pair.split()[0])
                    pairs_id[i, 1] = int(pair.split()[1])

                # Extract ground truth transformations
                all_idx = np.arange(len(info))
                gt_idx = np.delete(all_idx, pairs_id_idx)
                gt = np.array([line.strip().split() for line in np.array(info)[gt_idx]], dtype=np.float32)

                for pair in range(len(pairs_id)):
                    tgt_temp = os.path.join('test', scene, f"mesh_{pairs_id[pair][0]}.ply")
                    src_temp = os.path.join('test', scene, f"mesh_{pairs_id[pair][1]}.ply")
                    tgt.append(tgt_temp)
                    src.append(src_temp)

                    # Extract rotation and translation
                    gt_temp = gt[pair * 4: pair * 4 + 4, :]
                    rot.append(gt_temp[:3, :3])
                    trans.append(gt_temp[:3, 3])

                    # Compute overlap
                    src_pcd_path = os.path.join(self.data_path, 'geometry', src_temp)
                    tgt_pcd_path = os.path.join(self.data_path, 'geometry', tgt_temp)

                    if os.path.exists(src_pcd_path) and os.path.exists(tgt_pcd_path):
                        src_pcd = o3d.io.read_point_cloud(src_pcd_path)
                        tgt_pcd = o3d.io.read_point_cloud(tgt_pcd_path)
                        overlap_temp = compute_overlap_ratio(src_pcd, tgt_pcd, gt_temp, self.overlap_radius)
                        overlap.append(overlap_temp)
                    else:
                        print(f"Warning: Missing point cloud files for pair {src_temp}, {tgt_temp}")
                        overlap.append(0.0)

            # Save the '.pkl' file
            pkl_save = {
                "src": src,
                "tgt": tgt,
                "rot": rot,
                "trans": trans,
                "overlap": overlap
            }
            with open(os.path.join(pkl_save_path, f'{self.config_type}.pkl'), 'wb') as f:
                pickle.dump(pkl_save, f)

            print(f"\n'{os.path.join(pkl_save_path, f'{self.config_type}.pkl')}' is saved\n")
            self.pkl_info = pkl_save

    def process_dataset(self):
        """
        Process the IndoorLRS dataset by extracting features and performing multi-view fusion.
        """
        # Load pair information
        pkl_info = self.pkl_info
        pair_src_info = pkl_info['src'] if isinstance(pkl_info['src'], list) else pkl_info['src'].tolist()
        pair_tgt_info = pkl_info['tgt'] if isinstance(pkl_info['tgt'], list) else pkl_info['tgt'].tolist()

        # Remove redundant point cloud indices and sort by scene
        pcd_info_random = sorted(set(pair_src_info + pair_tgt_info))
        scene_all = ['apartment', 'bedroom', 'boardroom', 'lobby', 'loft']
        num_pcd = [320, 220, 244, 200, 253]
        pcd_info = []
        for scene in scene_all:
            pcd_scene = [item for item in pcd_info_random if scene in item]
            pcd_scene.sort(key=lambda y: int(re.findall(r'\d+', y)[0]))
            pcd_info.extend(pcd_scene)

        missing_ratio_all = []
        scene = None

        # Process each point cloud
        for pair_id, curr_pcd_dir in enumerate(tqdm(pcd_info, desc="Processing point clouds")):
            pcd_path_input = os.path.join(self.data_path, 'geometry', curr_pcd_dir)
            pcd_path_save = pcd_path_input.replace(self.input_root, self.output_root).replace('geometry', 'data')
            os.makedirs(os.path.dirname(pcd_path_save), exist_ok=True)

            # Skip already processed files
            if os.path.isfile(pcd_path_save.replace('.ply', '.npz')):
                continue

            # Load scene-specific data
            scene_curr = os.path.basename(os.path.dirname(curr_pcd_dir))
            if scene != scene_curr:
                scene = scene_curr
                num_pre = int(np.sum(num_pcd[:scene_all.index(scene_curr)]))
                image_list_rgb = sorted(os.listdir(os.path.join(self.data_path, 'image', scene, 'rgb')),
                                        key=lambda y: int(re.findall(r'\d+', y)[0]))
                image_list_depth = sorted(os.listdir(os.path.join(self.data_path, 'image', scene, 'depth')),
                                          key=lambda y: int(re.findall(r'\d+', y)[0]))
                image_pose_path = os.path.join(self.data_path, 'image', f'pose_{scene}.log')
                with open(image_pose_path, 'r') as f:
                    image_pose = f.readlines()
                fragment_pose_path = os.path.join(self.data_path, 'geometry', 'test',
                                                  os.path.basename(os.path.dirname(curr_pcd_dir)), 'pose_slac.log')
                with open(fragment_pose_path, 'r') as f:
                    fragment_pose = f.readlines()

            # Load raw geometry point cloud
            pcd_path_input = pcd_path_input.replace('all_test', 'test').replace('.npz', '.ply')
            if not os.path.isfile(pcd_path_input):
                print(f"Warning: Point cloud file not found: {pcd_path_input}")
                continue

            pcd_raw = o3d.io.read_point_cloud(pcd_path_input)

            # Use Open3D for downsampling
            pcd_downsampled = pcd_raw.voxel_down_sample(voxel_size=self.voxel_size)
            pcd_fragment = np.array(pcd_downsampled.points, dtype=np.float32)
            pcd_fragment_clr = np.array(pcd_downsampled.colors, dtype=np.float32)
            pcd_fragment = torch.from_numpy(np.asarray(pcd_fragment)).float()

            # Process based on the number of views
            if self.num_images == 3:
                data = self.three_view_img_process_indoorlrs(
                    scene,
                    image_list_rgb[3 * (pair_id - num_pre): 3 * ((pair_id - num_pre) + 1)],
                    image_list_depth[3 * (pair_id - num_pre): 3 * ((pair_id - num_pre) + 1)],
                    image_pose[5 * 3 * (pair_id - num_pre): 5 * 3 * ((pair_id - num_pre) + 1)]
                )
                pose_frame_1, pose_frame_2, pose_frame_3, color_frame_1, color_frame_2, color_frame_3, depth_frame_1, depth_frame_2, depth_frame_3, camera_intrinsic = data
                pcd_frames = self.pcd_fragment2frame(
                    fragment_pose[5 * (pair_id - num_pre): 5 * ((pair_id - num_pre) + 1)],
                    pcd_fragment, self.num_images, pose_frame_1, pose_frame_2, pose_frame_3
                )
                color_frames = self.three_view_img_transform(color_frame_1, color_frame_2, color_frame_3,
                                                             image_type='color')
                depth_frames = self.three_view_img_transform(depth_frame_1, depth_frame_2, depth_frame_3,
                                                             image_type='depth')

                image_size_before = np.array(color_frame_1.shape[:2]).tolist()
                # adjust camera intrinsic if the raw image size is not equal to the input of the 2D pretrianed weight
                camera_intrinsic_adjust = adjust_intrinsic(camera_intrinsic, image_size_before, self.image_size_after)

                pcd2img_frame_1, mask_1 = self.frame_pcd2image_projection(pcd_frames[0], depth_frames[0], camera_intrinsic_adjust)
                pcd2img_frame_2, mask_2 = self.frame_pcd2image_projection(pcd_frames[1], depth_frames[1], camera_intrinsic_adjust)
                pcd2img_frame_3, mask_3 = self.frame_pcd2image_projection(pcd_frames[2], depth_frames[2], camera_intrinsic_adjust)

                # Extract 2D image features
                for param in self.backbone2d.parameters():
                    param.requires_grad = False
                img_feats_2d_1 = self.backbone2d(color_frames[0].unsqueeze(0).to(self.device)).squeeze(0)
                img_feats_2d_2 = self.backbone2d(color_frames[1].unsqueeze(0).to(self.device)).squeeze(0)
                img_feats_2d_3 = self.backbone2d(color_frames[2].unsqueeze(0).to(self.device)).squeeze(0)

                img_feats_3d_1 = self.feats_idx_2d_to_3d(pcd2img_frame_1, img_feats_2d_1, mask_1)
                img_feats_3d_2 = self.feats_idx_2d_to_3d(pcd2img_frame_2, img_feats_2d_2, mask_2)
                img_feats_3d_3 = self.feats_idx_2d_to_3d(pcd2img_frame_3, img_feats_2d_3, mask_3)

                # fuse img feats from multi-view img
                img_feats_final, idx_img_feats_valid, missing_ratio = (
                    self.three_view_img_feat_fusion(pcd_fragment, mask_1, mask_2, mask_3, img_feats_3d_1,
                                                    img_feats_3d_2, img_feats_3d_3))
            elif self.num_images == 5:
                data = self.five_view_img_process_indoorlrs(
                    scene,
                    image_list_rgb[5 * (pair_id - num_pre): 5 * ((pair_id - num_pre) + 1)],
                    image_list_depth[5 * (pair_id - num_pre): 5 * ((pair_id - num_pre) + 1)],
                    image_pose[5 * 5 * (pair_id - num_pre): 5 * 5 * ((pair_id - num_pre) + 1)]
                )
                pose_frames, color_frames, depth_frames, camera_intrinsic = data[:5], data[5:10], data[10:15], data[15]
                pcd_frames = self.pcd_fragment2frame(
                    fragment_pose[5 * (pair_id - num_pre): 5 * ((pair_id - num_pre) + 1)],
                    pcd_fragment, self.num_images, *pose_frames
                )
                color_frames = self.five_view_img_transform(*color_frames, image_type='color')
                depth_frames = self.five_view_img_transform(*depth_frames, image_type='depth')

                image_size_before = np.array(color_frame_1.shape[:2]).tolist()
                # adjust camera intrinsic if the raw image size is not equal to the input of the 2D pretrianed weight
                camera_intrinsic_adjust = adjust_intrinsic(camera_intrinsic, image_size_before, self.image_size_after)

                pcd2img_frame_1, mask_1 = self.frame_pcd2image_projection(pcd_frames[0], depth_frames[0],
                                                                          camera_intrinsic_adjust)
                pcd2img_frame_2, mask_2 = self.frame_pcd2image_projection(pcd_frames[1], depth_frames[1],
                                                                          camera_intrinsic_adjust)
                pcd2img_frame_3, mask_3 = self.frame_pcd2image_projection(pcd_frames[2], depth_frames[2],
                                                                          camera_intrinsic_adjust)

                pcd2img_frame_4, mask_4 = self.frame_pcd2image_projection(pcd_frames[3], depth_frames[3],
                                                                          camera_intrinsic_adjust)
                pcd2img_frame_5, mask_5 = self.frame_pcd2image_projection(pcd_frames[4], depth_frames[4],
                                                                          camera_intrinsic_adjust)

                for param in self.backbone2d.parameters():
                    param.requires_grad = False
                img_feats_2d_1 = self.backbone2d(color_frames[0].unsqueeze(0).to(self.device)).squeeze(0)
                img_feats_2d_2 = self.backbone2d(color_frames[1].unsqueeze(0).to(self.device)).squeeze(0)
                img_feats_2d_3 = self.backbone2d(color_frames[2].unsqueeze(0).to(self.device)).squeeze(0)

                img_feats_2d_4 = self.backbone2d(color_frames[3].unsqueeze(0).to(self.device)).squeeze(0)
                img_feats_2d_5 = self.backbone2d(color_frames[4].unsqueeze(0).to(self.device)).squeeze(0)

                img_feats_3d_1 = self.feats_idx_2d_to_3d(pcd2img_frame_1, img_feats_2d_1, mask_1)
                img_feats_3d_2 = self.feats_idx_2d_to_3d(pcd2img_frame_2, img_feats_2d_2, mask_2)
                img_feats_3d_3 = self.feats_idx_2d_to_3d(pcd2img_frame_3, img_feats_2d_3, mask_3)

                img_feats_3d_4 = self.feats_idx_2d_to_3d(pcd2img_frame_4, img_feats_2d_4, mask_4)
                img_feats_3d_5 = self.feats_idx_2d_to_3d(pcd2img_frame_5, img_feats_2d_5, mask_5)

                # fuse img feats from multi-view img
                img_feats_final, idx_img_feats_valid = (
                    self.five_view_img_feat_fusion(pcd_fragment, mask_1, mask_2, mask_3, mask_4, mask_5, img_feats_3d_1,
                                                   img_feats_3d_2, img_feats_3d_3, img_feats_3d_4, img_feats_3d_5))

            # Save results
            np.savez_compressed(
                pcd_path_save.replace('.ply', ''),
                xyz=pcd_fragment,
                rgb=pcd_fragment_clr,
                img_feats=img_feats_final.cpu(),
                idx_img_feats_valid=idx_img_feats_valid.cpu()
            )
            missing_ratio_all.append(missing_ratio)

        # Save missing ratios
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(pcd_path_save))), 'missing_ratio.txt'),
                  'w') as file:
            for item in missing_ratio_all:
                file.write(f"{item}\n")

    def _process_views(self, pcd_frames, depth_frames, camera_intrinsic, color_frames):
        """
        Helper function to process views and extract features.
        """
        masks, img_feats_3d = [], []
        for pcd_frame, depth_frame, color_frame in zip(pcd_frames, depth_frames, color_frames):
            mask, pcd2img = self.frame_pcd2image_projection(pcd_frame, depth_frame, camera_intrinsic)
            img_feats_2d = self.backbone2d(color_frame.unsqueeze(0).to(self.device)).squeeze(0)
            img_feats_3d.append(self.feats_idx_2d_to_3d(pcd2img, img_feats_2d, mask))
            masks.append(mask)
        return masks, img_feats_3d

    def three_view_img_process_indoorlrs(self, scene, image_list_rgb, image_list_depth, image_pose):
        """
        Process three-view image data for the IndoorLRS dataset.
        """
        scene_path = os.path.join(self.data_path, 'image', scene)
        intrinsic_path = os.path.join(os.path.dirname(scene_path), "camera-intrinsics.txt")

        # Check if camera intrinsic file exists
        if not os.path.isfile(intrinsic_path):
            raise FileNotFoundError(f"Camera intrinsic file not found: {intrinsic_path}")
        camera_intrinsic = np.loadtxt(intrinsic_path)

        # Extract pose matrices
        pose_frame_1 = list2ndarray(image_pose[5 * 0 + 1: 5 * 1])
        pose_frame_2 = list2ndarray(image_pose[5 * 1 + 1: 5 * 2])
        pose_frame_3 = list2ndarray(image_pose[5 * 2 + 1: 5 * 3])

        # Load RGB and depth images
        color_frames = []
        depth_frames = []
        for i in range(3):
            color_path = os.path.join(scene_path, 'rgb', image_list_rgb[i])
            depth_path = os.path.join(scene_path, 'depth', image_list_depth[i])

            if not os.path.isfile(color_path) or not os.path.isfile(depth_path):
                raise FileNotFoundError(f"Image file not found: {color_path} or {depth_path}")

            color_frames.append(np.array(Image.open(color_path)))
            depth_frames.append(np.array(Image.open(depth_path)))

        return (pose_frame_1, pose_frame_2, pose_frame_3, *color_frames, *depth_frames, camera_intrinsic)

    def five_view_img_process_indoorlrs(self, scene, image_list_rgb, image_list_depth, image_pose):
        """
        Process five-view image data for the IndoorLRS dataset.
        """
        scene_path = os.path.join(self.data_path, 'image', scene)
        intrinsic_path = os.path.join(os.path.dirname(scene_path), "camera-intrinsics.txt")

        # Check if camera intrinsic file exists
        if not os.path.isfile(intrinsic_path):
            raise FileNotFoundError(f"Camera intrinsic file not found: {intrinsic_path}")
        camera_intrinsic = np.loadtxt(intrinsic_path)

        # Extract pose matrices
        pose_frames = [list2ndarray(image_pose[5 * i + 1: 5 * (i + 1)]) for i in range(5)]

        # Load RGB and depth images
        color_frames = []
        depth_frames = []
        for i in range(5):
            color_path = os.path.join(scene_path, 'rgb', image_list_rgb[i])
            depth_path = os.path.join(scene_path, 'depth', image_list_depth[i])

            if not os.path.isfile(color_path) or not os.path.isfile(depth_path):
                raise FileNotFoundError(f"Image file not found: {color_path} or {depth_path}")

            color_frames.append(np.array(Image.open(color_path)))
            depth_frames.append(np.array(Image.open(depth_path)))

        return (*pose_frames, *color_frames, *depth_frames, camera_intrinsic)

    def pcd_fragment2frame(self, fragment_pose, pcd_fragment, num_frame, pose_frame_1, pose_frame_2=None,
                           pose_frame_3=None, pose_frame_4=None, pose_frame_5=None, data_augmentation=False):
        """
        Transform fragment point cloud into frame-specific point clouds.
        """
        fragment_pose = list2ndarray(fragment_pose[1:])
        pcd_to_mid = array_transform(pcd_fragment, fragment_pose)

        def transform_pose(pose):
            pose[:3, 3] = np.matmul(pose[:3, :3].T, -pose[:3, 3])
            pose[:3, :3] = pose[:3, :3].T
            return array_transform(pcd_to_mid, pose)

        pcd_frame_1 = transform_pose(pose_frame_1)

        if num_frame == 3:
            pcd_frame_2 = transform_pose(pose_frame_2)
            pcd_frame_3 = transform_pose(pose_frame_3)
            return pcd_frame_1, pcd_frame_2, pcd_frame_3

        elif num_frame == 5:
            pcd_frame_2 = transform_pose(pose_frame_2)
            pcd_frame_3 = transform_pose(pose_frame_3)
            pcd_frame_4 = transform_pose(pose_frame_4)
            pcd_frame_5 = transform_pose(pose_frame_5)
            return pcd_frame_1, pcd_frame_2, pcd_frame_3, pcd_frame_4, pcd_frame_5

        else:
            print("Only use a single frame point cloud.")
            return pcd_frame_1