import torch
import os
import numpy as np
from tqdm import tqdm
import pickle
from PIL import Image
from datasets.base_extractor import Processor, adjust_intrinsic
import shutil


class ThreeDMatchDataset_Process(Processor):
    """
    Class for processing the 3DMatch dataset.
    Handles dataset processing, image feature extraction, and multi-view fusion.
    """

    def __init__(self, config):
        # Initialize the base Processor class
        super().__init__(config)

        # Process the dataset
        self.process_dataset()

    def single_view_img_process(self, curr_pcd_dir):
        """
        Process a single view image and its corresponding data.
        """
        pose_ref = os.path.join(self.data_path, 'geometry', curr_pcd_dir.replace('pth', 'info.txt'))
        with open(pose_ref) as f:
            info_ref = f.readlines()

        scene = os.path.basename(os.path.dirname(curr_pcd_dir))
        seq_ref = info_ref[0].split()[1]
        pose_id_ref_1 = info_ref[0].split()[2]

        scene_path = os.path.join(self.data_path, 'image', scene)
        camera_intrinsic = np.loadtxt(os.path.join(scene_path, "camera-intrinsics.txt"))

        filenames = [f"frame-{pose_id_ref_1.zfill(6)}"]
        pose_frame_1 = np.loadtxt(os.path.join(scene_path, seq_ref, f'{filenames[0]}.pose.txt'))

        # Load color and depth images
        color_path_ref_1 = os.path.join(scene_path, seq_ref, f'{filenames[0]}.color.png')
        if not os.path.exists(color_path_ref_1):
            color_path_ref_1 = os.path.join(scene_path, seq_ref, f'{filenames[0]}.color.jpg')
        color_frame_1 = np.array(Image.open(color_path_ref_1))

        depth_path_ref_1 = os.path.join(scene_path, seq_ref, f'{filenames[0]}.depth.png')
        if not os.path.exists(depth_path_ref_1):
            depth_path_ref_1 = os.path.join(scene_path, seq_ref, f'{filenames[0]}.depth.jpg')
        depth_frame_1 = np.array(Image.open(depth_path_ref_1))

        return pose_frame_1, color_frame_1, depth_frame_1, camera_intrinsic

    def three_view_img_process(self, curr_pcd_dir):
        """
        Process three-view images and their corresponding data.
        """
        pose_ref = os.path.join(self.data_path, 'geometry', curr_pcd_dir.replace('pth', 'info.txt'))
        with open(pose_ref) as f:
            info_ref = f.readlines()

        scene = os.path.basename(os.path.dirname(curr_pcd_dir))
        seq_ref = info_ref[0].split()[1]
        pose_ids = [
            info_ref[0].split()[2],
            info_ref[0].split()[3],
            str(int((int(info_ref[0].split()[2]) + int(info_ref[0].split()[3])) / 2))
        ]

        scene_path = os.path.join(self.data_path, 'image', scene)
        camera_intrinsic = np.loadtxt(os.path.join(scene_path, "camera-intrinsics.txt"))

        poses, color_frames, depth_frames = [], [], []
        for pose_id in pose_ids:
            filename = f"frame-{pose_id.zfill(6)}"
            poses.append(np.loadtxt(os.path.join(scene_path, seq_ref, f'{filename}.pose.txt')))

            # Load color image
            color_path = os.path.join(scene_path, seq_ref, f'{filename}.color.png')
            if not os.path.exists(color_path):
                color_path = os.path.join(scene_path, seq_ref, f'{filename}.color.jpg')
            color_frames.append(np.array(Image.open(color_path)))

            # Load depth image
            depth_path = os.path.join(scene_path, seq_ref, f'{filename}.depth.png')
            if not os.path.exists(depth_path):
                depth_path = os.path.join(scene_path, seq_ref, f'{filename}.depth.jpg')
            depth_frames.append(np.array(Image.open(depth_path)))

        return (*poses, *color_frames, *depth_frames, camera_intrinsic)

    def five_view_img_process(self, curr_pcd_dir):
        """
        Process five-view images and their corresponding data.
        """
        pose_ref = os.path.join(self.data_path, 'geometry', curr_pcd_dir.replace('pth', 'info.txt'))
        with open(pose_ref) as f:
            info_ref = f.readlines()

        scene = os.path.basename(os.path.dirname(curr_pcd_dir))
        seq_ref = info_ref[0].split()[1]
        pose_id_ref_1 = int(info_ref[0].split()[2])
        pose_id_ref_2 = int(info_ref[0].split()[3])
        pose_ids = [
            pose_id_ref_1,
            pose_id_ref_2,
            (pose_id_ref_1 + pose_id_ref_2) // 2,
            pose_id_ref_1 + (pose_id_ref_2 - pose_id_ref_1) // 4,
            pose_id_ref_1 + 3 * (pose_id_ref_2 - pose_id_ref_1) // 4
        ]

        scene_path = os.path.join(self.data_path, 'image', scene)
        camera_intrinsic = np.loadtxt(os.path.join(scene_path, "camera-intrinsics.txt"))

        poses, color_frames, depth_frames = [], [], []
        for pose_id in pose_ids:
            filename = f"frame-{str(pose_id).zfill(6)}"
            poses.append(np.loadtxt(os.path.join(scene_path, seq_ref, f'{filename}.pose.txt')))

            # Load color image
            color_path = os.path.join(scene_path, seq_ref, f'{filename}.color.png')
            if not os.path.exists(color_path):
                color_path = os.path.join(scene_path, seq_ref, f'{filename}.color.jpg')
            color_frames.append(np.array(Image.open(color_path)))

            # Load depth image
            depth_path = os.path.join(scene_path, seq_ref, f'{filename}.depth.png')
            if not os.path.exists(depth_path):
                depth_path = os.path.join(scene_path, seq_ref, f'{filename}.depth.jpg')
            depth_frames.append(np.array(Image.open(depth_path)))

        return (*poses, *color_frames, *depth_frames, camera_intrinsic)

    def pcd_fragment2frame(self, pcd_fragment, num_frame, pose_frame_1, pose_frame_2=None, pose_frame_3=None,
                           pose_frame_4=None, pose_frame_5=None, data_augmentation=False):
        """
        Transform a point cloud fragment into multiple frames based on given poses.

        Args:
            pcd_fragment (np.ndarray): Input point cloud fragment.
            num_frame (int): Number of frames (3 or 5).
            pose_frame_1, ..., pose_frame_5 (np.ndarray): Transformation matrices for each frame.
            data_augmentation (bool): Whether to apply data augmentation (not implemented).

        Returns:
            Transformed point cloud frames as torch.FloatTensor.
        """
        pos_ref_1_rot, pose_ref_1_tra = pose_frame_1[:3, :3], pose_frame_1[:3, 3]
        pcd_to_mid = np.matmul(pcd_fragment, pos_ref_1_rot.T) + pose_ref_1_tra
        pcd_frame_1 = pcd_fragment.clone().detach().float()

        if num_frame == 3:
            poses = [pose_frame_2, pose_frame_3]
        elif num_frame == 5:
            poses = [pose_frame_2, pose_frame_3, pose_frame_4, pose_frame_5]
        else:
            print('Only use a single frame pcd or unsupported number of frames.')
            return pcd_frame_1

        pcd_frames = [pcd_frame_1]
        for pose in poses:
            pos_rot, pos_tra = pose[:3, :3], pose[:3, 3]
            pcd_mid_to_frame_tra = pcd_to_mid - pos_tra
            pcd_frame = np.matmul(pcd_mid_to_frame_tra, pos_rot)
            pcd_frames.append(pcd_frame.clone().detach().float())

        return tuple(pcd_frames)

    def process_dataset(self):
        """
        Process the dataset by extracting image features and fusing them with point cloud data.
        """
        # Read pair information
        with open(os.path.join(self.data_path, 'metadata', f'{self.benchmark_name}_converted.pkl'), "rb") as f:
            pkl_info = pickle.load(f)

        pair_src_info = pkl_info['src']
        pair_tgt_info = pkl_info['tgt']
        pcd_info = sorted(list(set(pair_src_info + pair_tgt_info))) # Unique point cloud indices
        # pcd_info = sorted(set(pair_src_info.tolist() + pair_tgt_info.tolist()))  # Unique point cloud indices

        missing_ratio_all = []

        # Loop through all point cloud files
        for pair_id, curr_pcd_dir in enumerate(tqdm(pcd_info)):
            # Load raw point cloud data
            pcd_path_input = os.path.join(self.data_path, 'geometry', curr_pcd_dir)
            pcd_fragment = torch.tensor(torch.load(pcd_path_input, weights_only=False), dtype=torch.float32)

            # Save path for output
            pcd_path_save = pcd_path_input.replace(self.input_root, self.output_root).replace('geometry', 'data')
            pcd_path_save = pcd_path_save.replace('test',
                                                  f'{self.benchmark_name}_{self.fusion_type}_{self.num_images}/test')
            pcd_path_save = pcd_path_save.replace('train',
                                                  f'{self.benchmark_name}_{self.fusion_type}_{self.num_images}/train')
            os.makedirs(os.path.dirname(pcd_path_save), exist_ok=True)

            # Process based on the number of views
            if self.num_images in [1, 2, 3]:
                results = self.three_view_img_process(curr_pcd_dir)
                pose_frames, color_frames, depth_frames, camera_intrinsic = results[:3], results[3:6], results[6:9], \
                results[9]

                # Adjust camera intrinsic
                image_size_before = np.array(color_frames[0].shape[:2]).tolist()
                camera_intrinsic_adjust = adjust_intrinsic(camera_intrinsic, image_size_before, self.image_size_after)

                # Transform point cloud into frames
                pcd_frames = self.pcd_fragment2frame(pcd_fragment, 3, *pose_frames)

                # Resize color and depth images
                color_frames = self.three_view_img_transform(*color_frames, image_type='color')
                depth_frames = self.three_view_img_transform(*depth_frames, image_type='depth')

                # Project point cloud to image and extract 2D features
                pcd2img_frames, masks = [], []
                for i in range(3):
                    pcd2img, mask = self.frame_pcd2image_projection(pcd_frames[i], depth_frames[i],
                                                                    camera_intrinsic_adjust)
                    pcd2img_frames.append(pcd2img)
                    masks.append(mask)

                # Extract 2D image features
                img_feats_2d = [self.backbone2d(color_frame.unsqueeze(0).to(self.device)).squeeze(0) for color_frame in
                                color_frames]

                # Convert 2D features to 3D
                img_feats_3d = [self.feats_idx_2d_to_3d(pcd2img_frames[i], img_feats_2d[i], masks[i]) for i in range(3)]

                # Fuse image features
                if self.num_images == 1:
                    img_feats_final, idx_img_feats_valid, missing_ratio = self.single_view_img_feat(pcd_fragment,
                                                                                                    masks[0],
                                                                                                    img_feats_3d[0])
                elif self.num_images == 2:
                    img_feats_final, idx_img_feats_valid, missing_ratio = self.two_view_img_feat_fusion(
                        pcd_fragment, masks[0], masks[1], img_feats_3d[0], img_feats_3d[1])
                elif self.num_images == 3:
                    img_feats_final, idx_img_feats_valid, missing_ratio = self.three_view_img_feat_fusion(
                        pcd_fragment, masks[0], masks[1], masks[2], img_feats_3d[0], img_feats_3d[1], img_feats_3d[2])

            elif self.num_images in [4, 5]:
                results = self.five_view_img_process(curr_pcd_dir)
                pose_frames, color_frames, depth_frames, camera_intrinsic = results[:5], results[5:10], results[10:15], \
                results[15]

                # Adjust camera intrinsic
                image_size_before = np.array(color_frames[0].shape[:2]).tolist()
                camera_intrinsic_adjust = adjust_intrinsic(camera_intrinsic, image_size_before, self.image_size_after)

                # Transform point cloud into frames
                pcd_frames = self.pcd_fragment2frame(pcd_fragment, 5, *pose_frames)

                # Resize color and depth images
                color_frames = self.five_view_img_transform(*color_frames, image_type='color')
                depth_frames = self.five_view_img_transform(*depth_frames, image_type='depth')

                # Project point cloud to image and extract 2D features
                pcd2img_frames, masks = [], []
                for i in range(5):
                    pcd2img, mask = self.frame_pcd2image_projection(pcd_frames[i], depth_frames[i],
                                                                    camera_intrinsic_adjust)
                    pcd2img_frames.append(pcd2img)
                    masks.append(mask)

                # Extract 2D image features
                img_feats_2d = [self.backbone2d(color_frame.unsqueeze(0).to(self.device)).squeeze(0) for color_frame in
                                color_frames]

                # Convert 2D features to 3D
                img_feats_3d = [self.feats_idx_2d_to_3d(pcd2img_frames[i], img_feats_2d[i], masks[i]) for i in range(5)]

                # Fuse image features
                if self.num_images == 5:
                    img_feats_final, idx_img_feats_valid, missing_ratio = self.five_view_img_feat_fusion(
                        pcd_fragment, *masks, *img_feats_3d)
                elif self.num_images == 4:
                    img_feats_final, idx_img_feats_valid, missing_ratio = self.four_view_img_feat_fusion(
                        pcd_fragment, masks[0], masks[1], masks[3], masks[4], img_feats_3d[0], img_feats_3d[1],
                        img_feats_3d[3], img_feats_3d[4])

            # Save processed data
            np.savez_compressed(
                pcd_path_save.replace('.pth', ''),
                xyz=pcd_fragment,
                img_feats=img_feats_final.detach().cpu(),
                idx_img_feats_valid=idx_img_feats_valid.detach().cpu()
            )
            shutil.copy2(f'{os.path.splitext(pcd_path_input)[0]}.info.txt', os.path.dirname(pcd_path_save))

            missing_ratio_all.append(missing_ratio)

        # Save missing ratios
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(pcd_path_save))), 'missing_ratio.txt'),
                  'w') as file:
            for item in missing_ratio_all:
                file.write(str(item) + '\n')