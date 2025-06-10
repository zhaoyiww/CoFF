import torch
import os
import open3d as o3d
import os.path as osp
import numpy as np
from tqdm import tqdm
import pickle
from PIL import Image
import shutil
from utils.o3d_tools import array2pcd, tensor2pcd
from utils.visualization import draw_registration_result_three_views
import copy
from torchvision.transforms import transforms, InterpolationMode


def adjust_intrinsic(intrinsic_before, image_dim_before, image_dim_after):
    """
    Adjust camera intrinsics if the image is resized.
    """
    if image_dim_before == image_dim_after:
        return intrinsic_before

    intrinsic_return = np.copy(intrinsic_before)
    height_ratio = image_dim_after[0] / image_dim_before[0]
    width_ratio = image_dim_after[1] / image_dim_before[1]

    intrinsic_return[0, 0] *= width_ratio
    intrinsic_return[1, 1] *= height_ratio
    intrinsic_return[0, 2] *= (image_dim_after[1] - 1) / (image_dim_before[1] - 1)
    intrinsic_return[1, 2] *= (image_dim_after[0] - 1) / (image_dim_before[0] - 1)

    return intrinsic_return


class Processor:
    """
    Processor class for handling point cloud and image feature extraction.
    """

    def __init__(self, config):
        self.input_root = config.input_root
        self.output_root = config.output_root
        self.dataset = config.dataset
        self.img_feats_dim = config.img_feats_dim
        self.num_images = config.num_images
        self.benchmark_name = config.benchmark_name
        self.fusion_type = config.fusion_type
        self.device = config.device
        self.depth_threshold = config.depth_threshold
        self.use_threshold = True
        self.data_path = self.input_root

        self.backbone2d = config.backbone2d.to(self.device)

        # input size of 2D backbone
        self.image_resize = [240, 320]
        self.image_depth_resize = [120, 160]
        # # output size of 2D backbone
        self.image_size_after = [120, 160]

    def single_view_img_transform(self, color_1, image_type='color'):
        """
        Resize and transform a single view image.
        """
        img_transform = self._get_image_transform(image_type)
        if image_type == 'depth':
            return img_transform(color_1) / 1000.0  # Convert depth to meters
        return img_transform(color_1)

    def three_view_img_transform(self, color_1, color_2, color_3, image_type='color'):
        """
        Resize and transform three view images.
        """
        img_transform = self._get_image_transform(image_type)
        if image_type == 'depth':
            return (
                img_transform(color_1) / 1000.0,
                img_transform(color_2) / 1000.0,
                img_transform(color_3) / 1000.0,
            )  # Convert depth to meters
        return img_transform(color_1), img_transform(color_2), img_transform(color_3)

    def five_view_img_transform(self, color_1, color_2, color_3, color_4, color_5, image_type='color'):
        """
        Resize and transform five view images.
        """
        img_transform = self._get_image_transform(image_type)
        if image_type == 'depth':
            return (
                img_transform(color_1) / 1000.0,
                img_transform(color_2) / 1000.0,
                img_transform(color_3) / 1000.0,
                img_transform(color_4) / 1000.0,
                img_transform(color_5) / 1000.0,
            )  # Convert depth to meters
        return (
            img_transform(color_1),
            img_transform(color_2),
            img_transform(color_3),
            img_transform(color_4),
            img_transform(color_5),
        )

    def _get_image_transform(self, image_type):
        """
        Helper function to get the appropriate image transform.
        """
        if image_type == 'color':
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.image_resize, interpolation=InterpolationMode.BILINEAR),
            ])
        elif image_type == 'depth':
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.image_depth_resize, interpolation=InterpolationMode.NEAREST),
            ])

    def matrix_multiplication(self, matrix, points):
        """
        Perform matrix multiplication to transform 3D points using a transformation matrix.

        Args:
            matrix (torch.Tensor): Transformation matrix (4x4 or 3x3).
            points (torch.Tensor): 3D points (nx3).

        Returns:
            torch.Tensor: Transformed 3D points (nx3).
        """
        # Add a row of ones to the points for homogeneous coordinates
        points_homogeneous = torch.cat([points.t().to(self.device), torch.ones((1, points.shape[0]), device=self.device)])

        # If the matrix is 3x3, convert it to 4x4 by adding identity rows/columns
        if matrix.shape[0] == 3:
            mat = torch.eye(4, device=self.device)
            mat[:3, :3] = matrix
            matrix = mat

        # Perform matrix multiplication and return the transformed points
        transformed_points = torch.mm(matrix, points_homogeneous.to(matrix.dtype)).t()[:, :3]
        return transformed_points

    def frame_pcd2image_projection(self, frame_pcd, depth_frame, camera_intrinsic, pcd_fragment_clr=None):
        """
        Project frame point cloud to image and get valid image pixels corresponding to the frame point cloud.
        """
        img_feats_size = self.image_size_after
        fx, fy = camera_intrinsic[0, 0], camera_intrinsic[1, 1]
        cx, cy = camera_intrinsic[0, 2], camera_intrinsic[1, 2]

        frame_pcd = frame_pcd.to(self.device)
        img_2d_coord = self.matrix_multiplication(torch.tensor(camera_intrinsic), frame_pcd)
        projected_depth = img_2d_coord[:, 2]
        img_2d_coord = (img_2d_coord[:, :2] / projected_depth.repeat(2, 1).T[:, :]).long()

        mask_height = (img_2d_coord[:, 1] >= 0) & (img_2d_coord[:, 1] < img_feats_size[0])
        mask_width = (img_2d_coord[:, 0] >= 0) & (img_2d_coord[:, 0] < img_feats_size[1])
        mask = mask_width & mask_height

        depth_frame = depth_frame.squeeze(0).to(self.device)
        depth_from_depth_image = depth_frame[img_2d_coord[mask, 1], img_2d_coord[mask, 0]]
        if self.use_threshold:
            mask_depth_valid = torch.abs(
                frame_pcd[:, 2][mask] - depth_from_depth_image) < self.depth_threshold
            mask[torch.where(mask)[0]] = mask_depth_valid

            # mask &= (depth_from_depth_image > 0) & (depth_from_depth_image < self.depth_threshold)

        idx = img_2d_coord[:, [1, 0]]
        return idx, mask

    def feats_idx_2d_to_3d(self, img_frame, color_frame_feats, mask):
        """
        Convert 2D image features to 3D point cloud features.
        """
        dim = self.img_feats_dim
        feats_valid = img_frame[mask]
        img_feats_3d = torch.zeros(feats_valid.shape[0], dim)
        for i in range(feats_valid.shape[0]):
            img_feats_3d[i] = color_frame_feats[:, feats_valid[i, 0], feats_valid[i, 1]]
        return img_feats_3d.to(self.device)

    def single_view_img_feat(self, pcd_fragment, mask_1, img_feats_3d_1):
        """
        Fuse single view image features with the point cloud.
        """
        # Initialize index tracking for valid features
        idx_all = torch.zeros((pcd_fragment.shape[0], 1), dtype=torch.float32).to(self.device)
        idx_all[mask_1, 0] = 1
        num_idx = torch.sum(idx_all, axis=1)

        # Assign features directly (no need for fusion logic here)
        img_feats = img_feats_3d_1

        # Identify valid indices and handle missing features
        idx_img_feats_valid = torch.where(num_idx > 0)[0]
        img_feats_final = torch.nan_to_num(img_feats, nan=1.0)
        zero_rows = torch.all(img_feats_final == 0, dim=1)
        img_feats_final[zero_rows] = 1

        # Calculate and print the missing ratio
        missing_ratio = (1 - len(idx_img_feats_valid) / img_feats_final.shape[0])
        print(f'Missing ratio of pixel-wise img feats: {missing_ratio:.3f}')

        return img_feats_final, idx_img_feats_valid, missing_ratio

    def two_view_img_feat_fusion(self, pcd_fragment, mask_1, mask_2, img_feats_3d_1, img_feats_3d_2):
        """
        Fuse image features from two views.
        """
        idx_all = torch.zeros((pcd_fragment.shape[0], 2), dtype=torch.float32).to(self.device)
        idx_all[mask_1, 0] = 1
        idx_all[mask_2, 1] = 1
        num_idx = torch.sum(idx_all, axis=1)

        img_feats_final, idx_img_feats_valid, missing_ratio = self._fuse_features(pcd_fragment, num_idx, idx_all, [mask_1, mask_2], [img_feats_3d_1, img_feats_3d_2])
        return img_feats_final, idx_img_feats_valid, missing_ratio

    def three_view_img_feat_fusion(self, pcd_fragment, mask_1, mask_2, mask_3, img_feats_3d_1, img_feats_3d_2, img_feats_3d_3):
        """
        Fuse image features from three views.
        """
        idx_all = torch.zeros((pcd_fragment.shape[0], 3), dtype=torch.float32).to(self.device)
        idx_all[mask_1, 0] = 1
        idx_all[mask_2, 1] = 1
        idx_all[mask_3, 2] = 1
        num_idx = torch.sum(idx_all, axis=1)

        img_feats_final, idx_img_feats_valid, missing_ratio = self._fuse_features(pcd_fragment, num_idx, idx_all, [mask_1, mask_2, mask_3], [img_feats_3d_1, img_feats_3d_2, img_feats_3d_3])
        return img_feats_final, idx_img_feats_valid, missing_ratio

    def four_view_img_feat_fusion(self, pcd_fragment, mask_1, mask_2, mask_4, mask_5, img_feats_3d_1, img_feats_3d_2, img_feats_3d_4, img_feats_3d_5):
        """
        Fuse image features from four views.
        """
        idx_all = torch.zeros((pcd_fragment.shape[0], 4), dtype=torch.float32).to(self.device)
        idx_all[mask_1, 0] = 1
        idx_all[mask_2, 1] = 1
        idx_all[mask_4, 2] = 1
        idx_all[mask_5, 3] = 1
        num_idx = torch.sum(idx_all, axis=1)

        img_feats_final, idx_img_feats_valid, missing_ratio = self._fuse_features(pcd_fragment, num_idx, idx_all, [mask_1, mask_2, mask_4, mask_5], [img_feats_3d_1, img_feats_3d_2, img_feats_3d_4, img_feats_3d_5])
        return img_feats_final, idx_img_feats_valid, missing_ratio

    def five_view_img_feat_fusion(self, pcd_fragment, mask_1, mask_2, mask_3, mask_4, mask_5, img_feats_3d_1, img_feats_3d_2, img_feats_3d_3, img_feats_3d_4, img_feats_3d_5):
        """
        Fuse image features from five views.
        """
        idx_all = torch.zeros((pcd_fragment.shape[0], 5), dtype=torch.float32).to(self.device)
        idx_all[mask_1, 0] = 1
        idx_all[mask_2, 1] = 1
        idx_all[mask_3, 2] = 1
        idx_all[mask_4, 3] = 1
        idx_all[mask_5, 4] = 1
        num_idx = torch.sum(idx_all, axis=1)

        img_feats_final, idx_img_feats_valid, missing_ratio = self._fuse_features(pcd_fragment, num_idx, idx_all, [mask_1, mask_2, mask_3, mask_4, mask_5], [img_feats_3d_1, img_feats_3d_2, img_feats_3d_3, img_feats_3d_4, img_feats_3d_5])
        return img_feats_final, idx_img_feats_valid, missing_ratio

    def _fuse_features(self, pcd_fragment, num_idx, idx_all, masks, img_feats_list):
        """
        Helper function to fuse features based on the fusion type.
        """
        img_feats = torch.zeros(pcd_fragment.shape[0], self.img_feats_dim).to(self.device)

        if self.fusion_type == 'average':
            for mask, img_feats_3d in zip(masks, img_feats_list):
                img_feats[mask, :] += img_feats_3d
            img_feats /= num_idx[:, None]
        elif self.fusion_type == 'complement':
            for mask, img_feats_3d in reversed(list(zip(masks, img_feats_list))):
                img_feats[mask, :] = img_feats_3d
        elif self.fusion_type == 'random':
            img_feats_before_fuse = torch.zeros(pcd_fragment.shape[0], self.img_feats_dim * len(masks)).to(self.device)
            for i, (mask, img_feats_3d) in enumerate(zip(masks, img_feats_list)):
                img_feats_before_fuse[mask, i * self.img_feats_dim:(i + 1) * self.img_feats_dim] = img_feats_3d

            idx_multi_feats = torch.where(num_idx > 1)
            img_feats_random = self.random_select_one_index(idx_all[idx_multi_feats], img_feats_before_fuse[idx_multi_feats])
            img_feats[idx_multi_feats[0], :] = img_feats_random

        idx_img_feats_valid = torch.where(num_idx > 0)[0]
        img_feats_final = torch.nan_to_num(img_feats, nan=1.0)
        zero_rows = torch.all(img_feats_final == 0, dim=1)
        img_feats_final[zero_rows] = 1

        missing_ratio = (1 - len(idx_img_feats_valid) / img_feats_final.shape[0])
        print(f'Missing ratio of pixel-wise img feats: {missing_ratio:.3f} of {pcd_fragment.shape[0]}')

        return img_feats_final, idx_img_feats_valid, missing_ratio

    def random_select_one_index(self, indices_multiple, img_feats_multiple):
        """
        Randomly select one valid feature index for each point in the point cloud.

        Args:
            indices_multiple (torch.Tensor): Binary tensor indicating valid indices for each point.
            img_feats_multiple (torch.Tensor): Tensor containing concatenated image features.

        Returns:
            torch.Tensor: Randomly selected image features for each point.
        """
        # Initialize tensor to store randomly selected features
        img_feats_random = torch.zeros(indices_multiple.shape[0], self.img_feats_dim).to(self.device)

        # Iterate over each point
        for i, indices in enumerate(indices_multiple):
            # Find valid indices (where value is 1)
            valid_indices = torch.where(indices == 1)[0]

            # Randomly select one valid index
            random_idx = valid_indices[torch.randint(len(valid_indices), (1,))].item()

            # Extract the corresponding features
            img_feats_random[i, :] = img_feats_multiple[i, self.img_feats_dim * random_idx:
                                                           self.img_feats_dim * (random_idx + 1)]

        return img_feats_random