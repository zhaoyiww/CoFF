from functools import partial

import numpy as np
import torch, cv2, json

from src.backbone_3d.modules.ops import grid_subsample, radius_search
from utils.torch import build_dataloader
import os
import os.path as osp
import copy
from PIL import Image
from easydict import EasyDict as edict


# Stack mode utilities

def list2ndarray(image_pose):
    pose_frame_1 = np.array(image_pose)
    pose_frame_1 = [line.strip() for line in pose_frame_1]
    pose_frame_1 = np.array([line.split() for line in pose_frame_1])
    pose_frame_1 = pose_frame_1.astype(np.float32)
    return pose_frame_1

def array_transform(src_pts, est_transform):
    rotation, translation = est_transform[:3, :3], est_transform[:3, 3]
    src_pts = np.matmul(src_pts, rotation.T) + translation
    return src_pts


def load_poses(pose_dir):
    """Load 4x4 transformation matrices from separate pose files."""
    pose_dict = {}

    for filename in os.listdir(pose_dir):
        if filename.endswith("pose.txt"):
            # image_id = filename.replace("pose.txt", "")
            image_id = filename.split('.')[0].split('_')[2]
            pose_path = osp.join(pose_dir, filename)

            # 读取 4×4 变换矩阵
            pose_matrix = np.loadtxt(pose_path, skiprows=1).reshape(4, 4)

            pose_dict[f'frame_{str(image_id).zfill(3)}'] = pose_matrix

    return pose_dict


def pcd_fragment2frame(fragment_pose, pcd_fragment, num_frame, pose_frame_1, pose_frame_2=None,
                       pose_frame_3=None, pose_frame_4=None, pose_frame_5=None, data_augmentation=False):
    fragment_pose = list2ndarray(fragment_pose[1:])

    # !!!
    # for IndoorLRS dataset, the fragment pcd is not equal to the first frame pcd
    pcd_to_mid = array_transform(pcd_fragment, fragment_pose)

    pose_frame_1[:3, 3] = np.matmul(pose_frame_1[:3, :3].T, - pose_frame_1[:3, 3])
    pose_frame_1[:3, :3] = pose_frame_1[:3, :3].T
    pcd_frame_1 = array_transform(pcd_to_mid, pose_frame_1)
    if num_frame == 3:
        pose_frame_2[:3, 3] = np.matmul(pose_frame_2[:3, :3].T, - pose_frame_2[:3, 3])
        pose_frame_2[:3, :3] = pose_frame_2[:3, :3].T
        pcd_frame_2 = array_transform(pcd_to_mid, pose_frame_2)

        pose_frame_3[:3, 3] = np.matmul(pose_frame_3[:3, :3].T, - pose_frame_3[:3, 3])
        pose_frame_3[:3, :3] = pose_frame_3[:3, :3].T
        pcd_frame_3 = array_transform(pcd_to_mid, pose_frame_3)

        return pcd_frame_1, pcd_frame_2, pcd_frame_3
    elif num_frame == 5:
        pose_frame_2[:3, 3] = np.matmul(pose_frame_2[:3, :3].T, - pose_frame_2[:3, 3])
        pose_frame_2[:3, :3] = pose_frame_2[:3, :3].T
        pcd_frame_2 = array_transform(pcd_to_mid, pose_frame_2)

        pose_frame_3[:3, 3] = np.matmul(pose_frame_3[:3, :3].T, - pose_frame_3[:3, 3])
        pose_frame_3[:3, :3] = pose_frame_3[:3, :3].T
        pcd_frame_3 = array_transform(pcd_to_mid, pose_frame_3)

        pose_frame_4[:3, 3] = np.matmul(pose_frame_4[:3, :3].T, - pose_frame_4[:3, 3])
        pose_frame_4[:3, :3] = pose_frame_4[:3, :3].T
        pcd_frame_4 = array_transform(pcd_to_mid, pose_frame_4)

        pose_frame_5[:3, 3] = np.matmul(pose_frame_5[:3, :3].T, - pose_frame_5[:3, 3])
        pose_frame_5[:3, :3] = pose_frame_5[:3, :3].T
        pose_frame_5 = array_transform(pcd_to_mid, pose_frame_5)

        return pcd_frame_1, pcd_frame_2, pcd_frame_3, pcd_frame_4, pose_frame_5
    else:
        print('Only use a single frame pcd')
        return pcd_frame_1

def get_depth_intrinsics(rgb_intrinsics, res_rgb, res_depth):
    """ Get depth intrinsics if only rgb_intrinsics is provided """
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


def extract_img_patch(color, w, h, img_size=64):
    image = Image.fromarray(color[h[0]: h[1], w[0]: w[1]])
    # resample=Image.Resampling.NEAREST
    image = image.resize((img_size, img_size))
    image = np.array(image, dtype=np.float32) / 255.0
    return image


def compute_img_range(K, sample_tmp, color, neighbor_radius=None):
    # sample_tmp = compute_bounding_box(single_pts, radius=neighbor_radius)
    # x, y --> width, height
    sample_tmp[:, 0] = np.round(sample_tmp[:, 0] * K[0, 0] / sample_tmp[:, 2] + K[0, 2])
    sample_tmp[:, 1] = np.round(sample_tmp[:, 1] * K[1, 1] / sample_tmp[:, 2] + K[1, 2])

    # [-4.0574e-01, 1.5852e+00, -1.2538e-05]

    width = np.array([np.clip(torch.min(sample_tmp[:, 0]), 0, color.shape[1]),
                      np.clip(torch.max(sample_tmp[:, 0]), 0, color.shape[1])], dtype=np.int32)
    height = np.array([np.clip(torch.min(sample_tmp[:, 1]), 0, color.shape[0]),
                       np.clip(torch.max(sample_tmp[:, 1]), 0, color.shape[0])], dtype=np.int32)
    return width, height


def three_view_img_preprocessing(collated_dict, ref_pcd_dir):
    # ref
    pose_ref = osp.join('data', '3DMatch', 'data', ref_pcd_dir.replace('pth', 'info.txt'))
    f = open(pose_ref)
    info_ref = f.readlines()

    scene = collated_dict['metadata']['scene_name']
    seq_ref = info_ref[0].split()[1]
    pose_id_ref_1 = info_ref[0].split()[2]
    pose_id_ref_2 = info_ref[0].split()[3]
    pose_id_ref_3 = str(int((int(pose_id_ref_1) + int(pose_id_ref_2)) / 2))

    scene_path = osp.join('data', '3DMatch', 'image', scene)
    camera_intrinsic = np.loadtxt(osp.join(scene_path, "camera-intrinsics.txt"))

    filenames = [f"frame-{pose_id_ref_1.zfill(6)}", f"frame-{pose_id_ref_2.zfill(6)}",
                 f"frame-{pose_id_ref_3.zfill(6)}"]

    pose_ref_1 = np.loadtxt(osp.join(scene_path, seq_ref, f'{filenames[0]}.pose.txt'))
    pose_ref_2 = np.loadtxt(osp.join(scene_path, seq_ref, f'{filenames[1]}.pose.txt'))
    pose_ref_3 = np.loadtxt(osp.join(scene_path, seq_ref, f'{filenames[2]}.pose.txt'))

    # pts is aligned to frame 1, ref
    pos_ref_1_rot, pose_ref_1_tra = pose_ref_1[:3, :3], pose_ref_1[:3, 3]
    pos_ref_2_rot, pose_ref_2_tra = pose_ref_2[:3, :3], pose_ref_2[:3, 3]
    pos_ref_3_rot, pose_ref_3_tra = pose_ref_3[:3, :3], pose_ref_3[:3, 3]

    # load raw images, ref
    color_path_ref_1 = osp.join(scene_path, seq_ref, f'{filenames[0]}.color.png')
    if not osp.exists(color_path_ref_1):
        color_path_ref_1 = osp.join(scene_path, seq_ref, f'{filenames[0]}.color.jpg')
        color_path_ref_2 = osp.join(scene_path, seq_ref, f'{filenames[1]}.color.jpg')
        color_path_ref_3 = osp.join(scene_path, seq_ref, f'{filenames[2]}.color.jpg')
    else:
        color_path_ref_2 = osp.join(scene_path, seq_ref, f'{filenames[1]}.color.png')
        color_path_ref_3 = osp.join(scene_path, seq_ref, f'{filenames[2]}.color.png')
    color_ref_1 = np.array(Image.open(color_path_ref_1))
    color_ref_2 = np.array(Image.open(color_path_ref_2))
    color_ref_3 = np.array(Image.open(color_path_ref_3))

    return (pos_ref_1_rot, pose_ref_1_tra, pos_ref_2_rot, pose_ref_2_tra, pos_ref_3_rot, pose_ref_3_tra,
            color_ref_1, color_ref_2, color_ref_3, camera_intrinsic)


def extract_pose_cam2world_scannetpp(cfg, scene):
    file_path = osp.join(cfg.input_root, 'image', scene, 'pose_intrinsic_imu.json')
    with open(file_path, 'r') as file:
        data = json.load(file)

    pose_matrices = []
    intrinsic_matrices = []

    for frame, frame_data in data.items():
        pose = frame_data['pose']
        intrinsic = frame_data['intrinsic']
        # convert to 4x4 matrix
        pose_matrix = np.array(pose)
        # pose_matrices[frame] = pose_matrix
        pose_matrices.append(pose_matrix)
        intrinsic_matrix = np.array(intrinsic)
        # intrinsic_matrices[frame] = intrinsic_matrix
        intrinsic_matrices.append(intrinsic_matrix)
    return pose_matrices, intrinsic_matrices


def three_view_img_preprocessing_indoorlrs(collated_dict, image_list_rgb, image_pose):
    scene = collated_dict['metadata']['scene_name']
    # ref
    scene_path = osp.join('data', 'IndoorLRS', 'image', scene)

    camera_intrinsic = np.loadtxt(osp.join(osp.dirname(scene_path), "camera-intrinsics.txt"))

    pose_frame_1 = list2ndarray(image_pose[5 * 0 + 1: 5 * 1])
    pose_frame_2 = list2ndarray(image_pose[5 * 1 + 1: 5 * 2])
    pose_frame_3 = list2ndarray(image_pose[5 * 2 + 1: 5 * 3])

    # load raw images, ref
    color_path_ref_1 = osp.join(scene_path, 'rgb', image_list_rgb[0])
    color_path_ref_2 = osp.join(scene_path, 'rgb', image_list_rgb[1])
    color_path_ref_3 = osp.join(scene_path, 'rgb', image_list_rgb[2])

    color_frame_1 = np.array(Image.open(color_path_ref_1))
    color_frame_2 = np.array(Image.open(color_path_ref_2))
    color_frame_3 = np.array(Image.open(color_path_ref_3))

    return (pose_frame_1, pose_frame_2, pose_frame_3, color_frame_1, color_frame_2, color_frame_3,
            camera_intrinsic)


def three_view_img_preprocessing_scannetpp(collated_dict, ref_pcd_dir):
    # ref
    # pose_ref = osp.join(path_dir, 'ScanNetpp', 'data', ref_pcd_dir.replace('pth', 'pose.txt'))
    pose_ref = osp.join('data', 'ScanNetpp', 'data', ref_pcd_dir.replace('npz', 'pose.txt'))

    f = open(pose_ref)
    info_ref = f.readlines()

    scene = collated_dict['metadata']['scene_name']
    seq_ref = info_ref[0].split()[1]
    pose_id_ref_1 = info_ref[0].split()[2]
    pose_id_ref_2 = info_ref[0].split()[3]
    pose_id_ref_3 = str(int((int(pose_id_ref_1) + int(pose_id_ref_2)) / 2))

    scene_path = osp.join('data', 'ScanNetpp', 'image', scene)
    temp_dict = edict(dict())
    temp_dict.input_root = osp.join('data', 'ScanNetpp')

    pose_matrices, intrinsic_matrices = extract_pose_cam2world_scannetpp(temp_dict, scene)

    camera_intrinsic_1 = intrinsic_matrices[int(pose_id_ref_1)]
    camera_intrinsic_2 = intrinsic_matrices[int(pose_id_ref_2)]
    camera_intrinsic_3 = intrinsic_matrices[int(pose_id_ref_3)]
    # camera_intrinsic = np.loadtxt(osp.join(scene_path, "camera-intrinsics.txt"))

    filenames = [f"frame_{pose_id_ref_1.zfill(6)}", f"frame_{pose_id_ref_2.zfill(6)}",
                 f"frame_{pose_id_ref_3.zfill(6)}"]

    # pose_ref_1 = np.loadtxt(osp.join(scene_path, seq_ref, f'{filenames[0]}.pose.txt'))
    # pose_ref_2 = np.loadtxt(osp.join(scene_path, seq_ref, f'{filenames[1]}.pose.txt'))
    # pose_ref_3 = np.loadtxt(osp.join(scene_path, seq_ref, f'{filenames[2]}.pose.txt'))

    pose_frame_1 = pose_matrices[int(pose_id_ref_1)]
    pose_frame_2 = pose_matrices[int(pose_id_ref_2)]
    pose_frame_3 = pose_matrices[int(pose_id_ref_3)]

    # pts is aligned to frame 1, ref
    pos_ref_1_rot, pose_ref_1_tra = pose_frame_1[:3, :3], pose_frame_1[:3, 3]
    pos_ref_2_rot, pose_ref_2_tra = pose_frame_2[:3, :3], pose_frame_2[:3, 3]
    pos_ref_3_rot, pose_ref_3_tra = pose_frame_3[:3, :3], pose_frame_3[:3, 3]

    # load raw images, ref
    color_path_ref_1 = osp.join(scene_path, 'rgb', f'{filenames[0]}.png')
    if not osp.exists(color_path_ref_1):
        color_path_ref_1 = osp.join(scene_path, 'rgb', f'{filenames[0]}.jpg')
        color_path_ref_2 = osp.join(scene_path, 'rgb', f'{filenames[1]}.jpg')
        color_path_ref_3 = osp.join(scene_path, 'rgb', f'{filenames[2]}.jpg')
    else:
        color_path_ref_2 = osp.join(scene_path, 'rgb', f'{filenames[1]}.png')
        color_path_ref_3 = osp.join(scene_path, 'rgb', f'{filenames[2]}.png')
    color_ref_1 = np.array(Image.open(color_path_ref_1))
    color_ref_2 = np.array(Image.open(color_path_ref_2))
    color_ref_3 = np.array(Image.open(color_path_ref_3))

    return (pos_ref_1_rot, pose_ref_1_tra, pos_ref_2_rot, pose_ref_2_tra, pos_ref_3_rot, pose_ref_3_tra,
            color_ref_1, color_ref_2, color_ref_3, camera_intrinsic_1, camera_intrinsic_2, camera_intrinsic_3)


def precompute_data_stack_mode(points, lengths, num_stages, voxel_size, radius, neighbor_limits):
    assert num_stages == len(neighbor_limits)

    points_list = []
    lengths_list = []
    neighbors_list = []
    subsampling_list = []
    upsampling_list = []

    # grid subsampling
    for i in range(num_stages):
        if i > 0:
            points, lengths = grid_subsample(points, lengths, voxel_size=voxel_size)
        points_list.append(points)
        lengths_list.append(lengths)
        voxel_size *= 2

    # radius search
    for i in range(num_stages):
        cur_points = points_list[i]
        cur_lengths = lengths_list[i]

        neighbors = radius_search(
            cur_points,
            cur_points,
            cur_lengths,
            cur_lengths,
            radius,
            neighbor_limits[i],
        )
        neighbors_list.append(neighbors)

        if i < num_stages - 1:
            sub_points = points_list[i + 1]
            sub_lengths = lengths_list[i + 1]

            subsampling = radius_search(
                sub_points,
                cur_points,
                sub_lengths,
                cur_lengths,
                radius,
                neighbor_limits[i],
            )
            subsampling_list.append(subsampling)

            upsampling = radius_search(
                cur_points,
                sub_points,
                cur_lengths,
                sub_lengths,
                radius * 2,
                neighbor_limits[i + 1],
            )
            upsampling_list.append(upsampling)

        radius *= 2

    return {
        'points': points_list,
        'lengths': lengths_list,
        'neighbors': neighbors_list,
        'subsampling': subsampling_list,
        'upsampling': upsampling_list,
    }


def single_collate_fn_stack_mode(
    data_dicts, num_stages, voxel_size, search_radius, neighbor_limits, precompute_data=True
):
    r"""Collate function for single point cloud in stack mode.

    Points are organized in the following order: [P_1, ..., P_B].
    The correspondence indices are within each point cloud without accumulation.

    Args:
        data_dicts (List[Dict])
        num_stages (int)
        voxel_size (float)
        search_radius (float)
        neighbor_limits (List[int])
        precompute_data (bool=True)

    Returns:
        collated_dict (Dict)
    """
    batch_size = len(data_dicts)
    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    # handle special keys: feats, points, normals
    if 'normals' in collated_dict:
        normals = torch.cat(collated_dict.pop('normals'), dim=0)
    else:
        normals = None
    feats = torch.cat(collated_dict.pop('feats'), dim=0)
    points_list = collated_dict.pop('points')
    lengths = torch.LongTensor([points.shape[0] for points in points_list])
    points = torch.cat(points_list, dim=0)

    if batch_size == 1:
        # remove wrapping brackets if batch_size is 1
        for key, value in collated_dict.items():
            collated_dict[key] = value[0]

    if normals is not None:
        collated_dict['normals'] = normals
    collated_dict['features'] = feats
    if precompute_data:
        input_dict = precompute_data_stack_mode(points, lengths, num_stages, voxel_size, search_radius, neighbor_limits)
        collated_dict.update(input_dict)
    else:
        collated_dict['points'] = points
        collated_dict['lengths'] = lengths
    collated_dict['batch_size'] = batch_size

    return collated_dict


# not used
def registration_collate_fn_stack_mode(
    data_dicts, num_stages, voxel_size, search_radius, neighbor_limits, precompute_data=True
):
    r"""Collate function for registration in stack mode.

    Points are organized in the following order: [ref_1, ..., ref_B, src_1, ..., src_B].
    The correspondence indices are within each point cloud without accumulation.

    Args:
        data_dicts (List[Dict])
        num_stages (int)
        voxel_size (float)
        search_radius (float)
        neighbor_limits (List[int])
        precompute_data (bool)

    Returns:
        collated_dict (Dict)
    """
    batch_size = len(data_dicts)
    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    # handle special keys: [ref_feats, src_feats] -> feats, [ref_points, src_points] -> points, lengths
    feats = torch.cat(collated_dict.pop('ref_feats') + collated_dict.pop('src_feats'), dim=0)
    points_list = collated_dict.pop('ref_points') + collated_dict.pop('src_points')
    lengths = torch.LongTensor([points.shape[0] for points in points_list])
    points = torch.cat(points_list, dim=0)

    if batch_size == 1:
        # remove wrapping brackets if batch_size is 1
        for key, value in collated_dict.items():
            collated_dict[key] = value[0]

    collated_dict['features'] = feats
    if precompute_data:
        input_dict = precompute_data_stack_mode(points, lengths, num_stages, voxel_size, search_radius, neighbor_limits)
        collated_dict.update(input_dict)
    else:
        collated_dict['points'] = points
        collated_dict['lengths'] = lengths
    collated_dict['batch_size'] = batch_size

    ############################################3
    # add img patch
    ##########################################################
    # img patch extraction
    super_pts_coord = collated_dict['points'][-1]
    super_pts_length = collated_dict['lengths'][-1]
    super_pts_neighbor_coord = collated_dict['points'][-2]
    super_pts_neighbor_index = collated_dict['subsampling'][-1]

    ref_pcd_dir = collated_dict['metadata']['pcd0']
    src_pcd_dir = collated_dict['metadata']['pcd1']

    #####################################
    # input ref_pcd_dir
    # output
    (pos_ref_1_rot, pose_ref_1_tra, pos_ref_2_rot, pose_ref_2_tra, pos_ref_3_rot, pose_ref_3_tra,
     color_ref_1, color_ref_2, color_ref_3, camera_intrinsic) = three_view_img_preprocessing(collated_dict, ref_pcd_dir)

    (pos_src_1_rot, pose_src_1_tra, pos_src_2_rot, pose_src_2_tra, pos_src_3_rot, pose_src_3_tra,
     color_src_1, color_src_2, color_src_3, _) = three_view_img_preprocessing(collated_dict, src_pcd_dir)

    # if aug_src > 0.5:
    #     ref_points = np.matmul(ref_points, aug_rotation.T)
    #     rotation = np.matmul(aug_rotation, rotation)
    #     translation = np.matmul(aug_rotation, translation)
    # else:
    #     src_points = np.matmul(src_points, aug_rotation.T)
    #     rotation = np.matmul(rotation, aug_rotation.T)

    if data_dict['use_augmentation']:
        aug_rot = data_dict['aug_rotation']
        if data_dict['aug_src'] > 0.5:
            ref_1_rot = np.eye(4)
            ref_1_rot[:3, :3] = np.linalg.inv(aug_rot.T)
            ref_1_rot = torch.from_numpy(ref_1_rot).float()

            src_1_rot = torch.eye(4).float()
        else:
            src_1_rot = np.eye(4)
            src_1_rot[:3, :3] = np.linalg.inv(aug_rot.T)
            src_1_rot = torch.from_numpy(src_1_rot).float()

            ref_1_rot = torch.eye(4).float()
    else:
        ref_1_rot = torch.eye(4).float()
        src_1_rot = torch.eye(4).float()

    import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(color_ref_1)
    # plt.show()

    img_patch_all, img_patch_length_all, img_patch_idx_all = [], [], []

    img_patch_length = copy.deepcopy(super_pts_length)
    img_patch_idx = torch.arange(super_pts_coord.shape[0])
    # img_patch_idx_2 = copy.deepcopy(img_patch_idx)
    outlier_idx = []

    # set img_size and minimum img_size
    img_size = 64
    min_size = 10
    # all pts, including ref and src
    for i in range(super_pts_coord.shape[0]):
        # get indices of neighbors of i, removing fake indices
        idx_temp = super_pts_neighbor_index[i, :][super_pts_neighbor_index[i, :] < super_pts_neighbor_coord.shape[0]]
        neigh_i = super_pts_neighbor_coord[idx_temp, :]

        # ref_1_rot = torch.eye(4).float()
        # src_1_rot = torch.eye(4).float()

        # ref
        if i < super_pts_length[0]:
            # neigh_i_raw = copy.deepcopy(neigh_i)
            # recover the raw pcd before data augmentation
            neigh_i_raw = np.matmul(neigh_i, ref_1_rot[:3, :3])
            pts_to_mid = np.matmul(neigh_i_raw, pos_ref_1_rot.T) + pose_ref_1_tra
            pts_to_frame_1 = copy.deepcopy(neigh_i_raw)
            # pts_to_frame_1 = np.matmul(neigh_i, ref_1_rot[:3, :3])

            # input camera intrinsics, a list of neigh_i pts, RGB image
            # output the width and height of each patch: (i, 64, 64, 3)
            width_1, height_1 = compute_img_range(camera_intrinsic, pts_to_frame_1, color_ref_1)
            img_size_1 = [width_1[1] - width_1[0], height_1[1] - height_1[0]]

            pts_to_frame_2_tra = pts_to_mid - pose_ref_2_tra
            pts_to_frame_2 = np.matmul(pts_to_frame_2_tra, pos_ref_2_rot)
            width_2, height_2 = compute_img_range(camera_intrinsic, pts_to_frame_2, color_ref_2)
            img_size_2 = [width_2[1] - width_2[0], height_2[1] - height_2[0]]

            pts_to_frame_3_tra = pts_to_mid - pose_ref_3_tra
            pts_to_frame_3 = np.matmul(pts_to_frame_3_tra, pos_ref_3_rot)
            width_3, height_3 = compute_img_range(camera_intrinsic, pts_to_frame_3, color_ref_3)
            img_size_3 = [width_3[1] - width_3[0], height_3[1] - height_3[0]]

            # mask_1 = (img_size_1[0] >= img_size_2[0]) & (img_size_1[1] >= img_size_2[1])
            # mask_2 = (img_size_1[0] >= img_size_3[0]) & (img_size_1[1] >= img_size_3[1])
            # mask_3 = (img_size_2[0] >= img_size_3[0]) & (img_size_2[1] >= img_size_3[1])

            mask_1_ = np.sum(img_size_1) >= np.sum(img_size_2)
            mask_2_ = np.sum(img_size_1) >= np.sum(img_size_3)
            mask_3_ = np.sum(img_size_2) >= np.sum(img_size_3)

            # print(ref_pcd_dir, i)
            img_patch = []

            # if ref_pcd_dir != 'train/sun3d-mit_76_417-76-417b_4/cloud_bin_163.pth' or i != 82:
            # if ref_pcd_dir != 'train/sun3d-harvard_c3-hv_c3_1/cloud_bin_23.pth' or i != 44:
            #     continue

            # mask_1 = (width_1[1] - width_1[0] > img_size) & (height_1[1] - height_1[0] > img_size)
            if all(num >= min_size for num in img_size_1) and mask_1_ and mask_2_:
                img_patch = extract_img_patch(color_ref_1, width_1, height_1, img_size=img_size)
                # img_patch_2 = extract_img_patch(color_ref_2, width_2, height_2, img_size=64)
                # img_patch_3 = extract_img_patch(color_ref_3, width_3, height_3, img_size=64)
            elif all(num >= min_size for num in img_size_2) and not mask_1_ and mask_3_:
                # img_patch_1 = extract_img_patch(color_ref_1, width_1, height_1, img_size=64)
                img_patch = extract_img_patch(color_ref_2, width_2, height_2, img_size=img_size)
                # img_patch_3 = extract_img_patch(color_ref_3, width_3, height_3, img_size=64)
            elif all(num >= min_size for num in img_size_3):
                # img_patch_1 = extract_img_patch(color_ref_1, width_1, height_1, img_size=64)
                # img_patch_2 = extract_img_patch(color_ref_2, width_2, height_2, img_size=64)
                img_patch = extract_img_patch(color_ref_3, width_3, height_3, img_size=img_size)
            else:
                img_patch_length[0] -= 1
                # remove certain index
                outlier_idx.append(i)
                # img_patch_idx_2 = torch.cat((img_patch_idx[:i], img_patch_idx[i + 1:]))

            # plt.figure()
            # plt.imshow(img_patch)
            # plt.show()
        # src
        else:
            neigh_i_raw = np.matmul(neigh_i, src_1_rot[:3, :3])

            pts_to_mid = np.matmul(neigh_i_raw, pos_src_1_rot.T) + pose_src_1_tra
            pts_to_frame_1 = np.matmul(neigh_i, src_1_rot[:3, :3])
            # pts_to_frame_1 = copy.deepcopy(neigh_i)

            # input camera intrinsics, a list of neigh_i pts, RGB image
            # output the width and height of each patch: (i, 64, 64, 3)
            width_1, height_1 = compute_img_range(camera_intrinsic, pts_to_frame_1, color_src_1)
            img_size_1 = [width_1[1] - width_1[0], height_1[1] - height_1[0]]

            pts_to_frame_2_tra = pts_to_mid - pose_src_2_tra
            pts_to_frame_2 = np.matmul(pts_to_frame_2_tra, pos_src_2_rot)
            width_2, height_2 = compute_img_range(camera_intrinsic, pts_to_frame_2, color_src_2)
            img_size_2 = [width_2[1] - width_2[0], height_2[1] - height_2[0]]

            pts_to_frame_3_tra = pts_to_mid - pose_src_3_tra
            pts_to_frame_3 = np.matmul(pts_to_frame_3_tra, pos_src_3_rot)
            width_3, height_3 = compute_img_range(camera_intrinsic, pts_to_frame_3, color_src_3)
            img_size_3 = [width_3[1] - width_3[0], height_3[1] - height_3[0]]

            mask_1_ = np.sum(img_size_1) >= np.sum(img_size_2)
            mask_2_ = np.sum(img_size_1) >= np.sum(img_size_3)
            mask_3_ = np.sum(img_size_2) >= np.sum(img_size_3)

            # print(src_pcd_dir, i)
            img_patch = []

            # mask_1 = (width_1[1] - width_1[0] > img_size) & (height_1[1] - height_1[0] > img_size)
            if all(num >= min_size for num in img_size_1) and mask_1_ and mask_2_:
                img_patch = extract_img_patch(color_src_1, width_1, height_1, img_size=img_size)
                # img_patch_2 = extract_img_patch(color_ref_2, width_2, height_2, img_size=64)
                # img_patch_3 = extract_img_patch(color_ref_3, width_3, height_3, img_size=64)
            elif all(num >= min_size for num in img_size_2) and not mask_1_ and mask_3_:
                # img_patch_1 = extract_img_patch(color_ref_1, width_1, height_1, img_size=64)
                img_patch = extract_img_patch(color_src_2, width_2, height_2, img_size=img_size)
                # img_patch_3 = extract_img_patch(color_ref_3, width_3, height_3, img_size=64)
            elif all(num >= min_size for num in img_size_3):
                # img_patch_1 = extract_img_patch(color_ref_1, width_1, height_1, img_size=64)
                # img_patch_2 = extract_img_patch(color_ref_2, width_2, height_2, img_size=64)
                img_patch = extract_img_patch(color_src_3, width_3, height_3, img_size=img_size)
            else:
                img_patch_length[1] -= 1
                # remove certain index
                # img_patch_idx_2 = torch.cat((img_patch_idx[:i], img_patch_idx[i + 1:]))
                outlier_idx.append(i)

        # print(f'neighs: {neigh_i.shape}')
        if img_patch != []:
            img_patch_all.append(img_patch)
        # img_patch_length_all.append(img_patch_length)
        # img_patch_idx_all.append(img_patch_idx)
    collated_dict['img_patch'] = img_patch_all
    collated_dict['img_patch_length'] = img_patch_length

    img_patch_idx = np.delete(img_patch_idx, outlier_idx)
    collated_dict['img_patch_idx'] = img_patch_idx
    ###################################################

    return collated_dict

def registration_collate_fn_stack_mode_3dmatch(
    data_dicts, num_stages, voxel_size, search_radius, neighbor_limits, precompute_data=True
):
    r"""Collate function for registration in stack mode.

    Points are organized in the following order: [ref_1, ..., ref_B, src_1, ..., src_B].
    The correspondence indices are within each point cloud without accumulation.

    Args:
        data_dicts (List[Dict])
        num_stages (int)
        voxel_size (float)
        search_radius (float)
        neighbor_limits (List[int])
        precompute_data (bool)

    Returns:
        collated_dict (Dict)
    """
    batch_size = len(data_dicts)
    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    # handle special keys: [ref_feats, src_feats] -> feats, [ref_points, src_points] -> points, lengths
    feats = torch.cat(collated_dict.pop('ref_feats') + collated_dict.pop('src_feats'), dim=0)
    points_list = collated_dict.pop('ref_points') + collated_dict.pop('src_points')
    lengths = torch.LongTensor([points.shape[0] for points in points_list])
    points = torch.cat(points_list, dim=0)

    if batch_size == 1:
        # remove wrapping brackets if batch_size is 1
        for key, value in collated_dict.items():
            collated_dict[key] = value[0]

    collated_dict['features'] = feats
    if precompute_data:
        input_dict = precompute_data_stack_mode(points, lengths, num_stages, voxel_size, search_radius, neighbor_limits)
        collated_dict.update(input_dict)
    else:
        collated_dict['points'] = points
        collated_dict['lengths'] = lengths
    collated_dict['batch_size'] = batch_size

    ############################################3
    # add img patch
    ##########################################################
    # img patch extraction
    super_pts_coord = collated_dict['points'][-1]
    super_pts_length = collated_dict['lengths'][-1]
    super_pts_neighbor_coord = collated_dict['points'][-2]
    super_pts_neighbor_index = collated_dict['subsampling'][-1]

    ref_pcd_dir = collated_dict['metadata']['pcd0']
    src_pcd_dir = collated_dict['metadata']['pcd1']

    #####################################
    # input ref_pcd_dir
    # output
    (pos_ref_1_rot, pose_ref_1_tra, pos_ref_2_rot, pose_ref_2_tra, pos_ref_3_rot, pose_ref_3_tra,
     color_ref_1, color_ref_2, color_ref_3, camera_intrinsic) = three_view_img_preprocessing(collated_dict, ref_pcd_dir)

    (pos_src_1_rot, pose_src_1_tra, pos_src_2_rot, pose_src_2_tra, pos_src_3_rot, pose_src_3_tra,
     color_src_1, color_src_2, color_src_3, _) = three_view_img_preprocessing(collated_dict, src_pcd_dir)

    # if aug_src > 0.5:
    #     ref_points = np.matmul(ref_points, aug_rotation.T)
    #     rotation = np.matmul(aug_rotation, rotation)
    #     translation = np.matmul(aug_rotation, translation)
    # else:
    #     src_points = np.matmul(src_points, aug_rotation.T)
    #     rotation = np.matmul(rotation, aug_rotation.T)

    if data_dict['use_augmentation']:
        aug_rot = data_dict['aug_rotation']
        if data_dict['aug_src'] > 0.5:
            ref_1_rot = np.eye(4)
            ref_1_rot[:3, :3] = np.linalg.inv(aug_rot.T)
            ref_1_rot = torch.from_numpy(ref_1_rot).float()

            src_1_rot = torch.eye(4).float()
        else:
            src_1_rot = np.eye(4)
            src_1_rot[:3, :3] = np.linalg.inv(aug_rot.T)
            src_1_rot = torch.from_numpy(src_1_rot).float()

            ref_1_rot = torch.eye(4).float()
    else:
        ref_1_rot = torch.eye(4).float()
        src_1_rot = torch.eye(4).float()

    import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(color_ref_1)
    # plt.show()

    img_patch_all, img_patch_length_all, img_patch_idx_all = [], [], []

    img_patch_length = copy.deepcopy(super_pts_length)
    img_patch_idx = torch.arange(super_pts_coord.shape[0])
    # img_patch_idx_2 = copy.deepcopy(img_patch_idx)
    outlier_idx = []

    # set img_size and minimum img_size
    img_size = 64
    min_size = 10
    # all pts, including ref and src
    for i in range(super_pts_coord.shape[0]):
        # get indices of neighbors of i, removing fake indices
        idx_temp = super_pts_neighbor_index[i, :][super_pts_neighbor_index[i, :] < super_pts_neighbor_coord.shape[0]]
        neigh_i = super_pts_neighbor_coord[idx_temp, :]

        # ref_1_rot = torch.eye(4).float()
        # src_1_rot = torch.eye(4).float()

        # ref
        if i < super_pts_length[0]:
            # neigh_i_raw = copy.deepcopy(neigh_i)
            # recover the raw pcd before data augmentation
            neigh_i_raw = np.matmul(neigh_i, ref_1_rot[:3, :3])
            pts_to_mid = np.matmul(neigh_i_raw, pos_ref_1_rot.T) + pose_ref_1_tra
            pts_to_frame_1 = copy.deepcopy(neigh_i_raw)
            # pts_to_frame_1 = np.matmul(neigh_i, ref_1_rot[:3, :3])

            # input camera intrinsics, a list of neigh_i pts, RGB image
            # output the width and height of each patch: (i, 64, 64, 3)
            width_1, height_1 = compute_img_range(camera_intrinsic, pts_to_frame_1, color_ref_1)
            img_size_1 = [width_1[1] - width_1[0], height_1[1] - height_1[0]]

            pts_to_frame_2_tra = pts_to_mid - pose_ref_2_tra
            pts_to_frame_2 = np.matmul(pts_to_frame_2_tra, pos_ref_2_rot)
            width_2, height_2 = compute_img_range(camera_intrinsic, pts_to_frame_2, color_ref_2)
            img_size_2 = [width_2[1] - width_2[0], height_2[1] - height_2[0]]

            pts_to_frame_3_tra = pts_to_mid - pose_ref_3_tra
            pts_to_frame_3 = np.matmul(pts_to_frame_3_tra, pos_ref_3_rot)
            width_3, height_3 = compute_img_range(camera_intrinsic, pts_to_frame_3, color_ref_3)
            img_size_3 = [width_3[1] - width_3[0], height_3[1] - height_3[0]]

            # mask_1 = (img_size_1[0] >= img_size_2[0]) & (img_size_1[1] >= img_size_2[1])
            # mask_2 = (img_size_1[0] >= img_size_3[0]) & (img_size_1[1] >= img_size_3[1])
            # mask_3 = (img_size_2[0] >= img_size_3[0]) & (img_size_2[1] >= img_size_3[1])

            mask_1_ = np.sum(img_size_1) >= np.sum(img_size_2)
            mask_2_ = np.sum(img_size_1) >= np.sum(img_size_3)
            mask_3_ = np.sum(img_size_2) >= np.sum(img_size_3)

            # print(ref_pcd_dir, i)
            img_patch = []

            # if ref_pcd_dir != 'train/sun3d-mit_76_417-76-417b_4/cloud_bin_163.pth' or i != 82:
            # if ref_pcd_dir != 'train/sun3d-harvard_c3-hv_c3_1/cloud_bin_23.pth' or i != 44:
            #     continue

            # mask_1 = (width_1[1] - width_1[0] > img_size) & (height_1[1] - height_1[0] > img_size)
            if all(num >= min_size for num in img_size_1) and mask_1_ and mask_2_:
                img_patch = extract_img_patch(color_ref_1, width_1, height_1, img_size=img_size)
                # img_patch_2 = extract_img_patch(color_ref_2, width_2, height_2, img_size=64)
                # img_patch_3 = extract_img_patch(color_ref_3, width_3, height_3, img_size=64)
            elif all(num >= min_size for num in img_size_2) and not mask_1_ and mask_3_:
                # img_patch_1 = extract_img_patch(color_ref_1, width_1, height_1, img_size=64)
                img_patch = extract_img_patch(color_ref_2, width_2, height_2, img_size=img_size)
                # img_patch_3 = extract_img_patch(color_ref_3, width_3, height_3, img_size=64)
            elif all(num >= min_size for num in img_size_3):
                # img_patch_1 = extract_img_patch(color_ref_1, width_1, height_1, img_size=64)
                # img_patch_2 = extract_img_patch(color_ref_2, width_2, height_2, img_size=64)
                img_patch = extract_img_patch(color_ref_3, width_3, height_3, img_size=img_size)
            else:
                img_patch_length[0] -= 1
                # remove certain index
                outlier_idx.append(i)
                # img_patch_idx_2 = torch.cat((img_patch_idx[:i], img_patch_idx[i + 1:]))

            # plt.figure()
            # plt.imshow(img_patch)
            # plt.show()
        # src
        else:
            neigh_i_raw = np.matmul(neigh_i, src_1_rot[:3, :3])

            pts_to_mid = np.matmul(neigh_i_raw, pos_src_1_rot.T) + pose_src_1_tra
            pts_to_frame_1 = np.matmul(neigh_i, src_1_rot[:3, :3])
            # pts_to_frame_1 = copy.deepcopy(neigh_i)

            # input camera intrinsics, a list of neigh_i pts, RGB image
            # output the width and height of each patch: (i, 64, 64, 3)
            width_1, height_1 = compute_img_range(camera_intrinsic, pts_to_frame_1, color_src_1)
            img_size_1 = [width_1[1] - width_1[0], height_1[1] - height_1[0]]

            pts_to_frame_2_tra = pts_to_mid - pose_src_2_tra
            pts_to_frame_2 = np.matmul(pts_to_frame_2_tra, pos_src_2_rot)
            width_2, height_2 = compute_img_range(camera_intrinsic, pts_to_frame_2, color_src_2)
            img_size_2 = [width_2[1] - width_2[0], height_2[1] - height_2[0]]

            pts_to_frame_3_tra = pts_to_mid - pose_src_3_tra
            pts_to_frame_3 = np.matmul(pts_to_frame_3_tra, pos_src_3_rot)
            width_3, height_3 = compute_img_range(camera_intrinsic, pts_to_frame_3, color_src_3)
            img_size_3 = [width_3[1] - width_3[0], height_3[1] - height_3[0]]

            mask_1_ = np.sum(img_size_1) >= np.sum(img_size_2)
            mask_2_ = np.sum(img_size_1) >= np.sum(img_size_3)
            mask_3_ = np.sum(img_size_2) >= np.sum(img_size_3)

            # print(src_pcd_dir, i)
            img_patch = []

            # mask_1 = (width_1[1] - width_1[0] > img_size) & (height_1[1] - height_1[0] > img_size)
            if all(num >= min_size for num in img_size_1) and mask_1_ and mask_2_:
                img_patch = extract_img_patch(color_src_1, width_1, height_1, img_size=img_size)
                # img_patch_2 = extract_img_patch(color_ref_2, width_2, height_2, img_size=64)
                # img_patch_3 = extract_img_patch(color_ref_3, width_3, height_3, img_size=64)
            elif all(num >= min_size for num in img_size_2) and not mask_1_ and mask_3_:
                # img_patch_1 = extract_img_patch(color_ref_1, width_1, height_1, img_size=64)
                img_patch = extract_img_patch(color_src_2, width_2, height_2, img_size=img_size)
                # img_patch_3 = extract_img_patch(color_ref_3, width_3, height_3, img_size=64)
            elif all(num >= min_size for num in img_size_3):
                # img_patch_1 = extract_img_patch(color_ref_1, width_1, height_1, img_size=64)
                # img_patch_2 = extract_img_patch(color_ref_2, width_2, height_2, img_size=64)
                img_patch = extract_img_patch(color_src_3, width_3, height_3, img_size=img_size)
            else:
                img_patch_length[1] -= 1
                # remove certain index
                # img_patch_idx_2 = torch.cat((img_patch_idx[:i], img_patch_idx[i + 1:]))
                outlier_idx.append(i)

        # print(f'neighs: {neigh_i.shape}')
        if img_patch != []:
            img_patch_all.append(img_patch)
        # img_patch_length_all.append(img_patch_length)
        # img_patch_idx_all.append(img_patch_idx)
    collated_dict['img_patch'] = img_patch_all
    collated_dict['img_patch_length'] = img_patch_length

    img_patch_idx = np.delete(img_patch_idx, outlier_idx)
    collated_dict['img_patch_idx'] = img_patch_idx
    ###################################################

    return collated_dict


def registration_collate_fn_stack_mode_indoorlrs(
    data_dicts, num_stages, voxel_size, search_radius, neighbor_limits, precompute_data=True
):
    r"""Collate function for registration in stack mode.

    Points are organized in the following order: [ref_1, ..., ref_B, src_1, ..., src_B].
    The correspondence indices are within each point cloud without accumulation.

    Args:
        data_dicts (List[Dict])
        num_stages (int)
        voxel_size (float)
        search_radius (float)
        neighbor_limits (List[int])
        precompute_data (bool)

    Returns:
        collated_dict (Dict)
    """
    batch_size = len(data_dicts)
    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    # handle special keys: [ref_feats, src_feats] -> feats, [ref_points, src_points] -> points, lengths
    feats = torch.cat(collated_dict.pop('ref_feats') + collated_dict.pop('src_feats'), dim=0)
    points_list = collated_dict.pop('ref_points') + collated_dict.pop('src_points')
    lengths = torch.LongTensor([points.shape[0] for points in points_list])
    points = torch.cat(points_list, dim=0)

    if batch_size == 1:
        # remove wrapping brackets if batch_size is 1
        for key, value in collated_dict.items():
            collated_dict[key] = value[0]

    collated_dict['features'] = feats
    if precompute_data:
        input_dict = precompute_data_stack_mode(points, lengths, num_stages, voxel_size, search_radius, neighbor_limits)
        collated_dict.update(input_dict)
    else:
        collated_dict['points'] = points
        collated_dict['lengths'] = lengths
    collated_dict['batch_size'] = batch_size

    ############################################3
    # add img patch
    ##########################################################
    # img patch extraction
    super_pts_coord = collated_dict['points'][-1]
    super_pts_length = collated_dict['lengths'][-1]
    super_pts_neighbor_coord = collated_dict['points'][-2]
    super_pts_neighbor_index = collated_dict['subsampling'][-1]

    # ref_pcd_dir = collated_dict['metadata']['pcd0']
    # src_pcd_dir = collated_dict['metadata']['pcd1']

    # path_dir = '../../data'
    scene = collated_dict['scene_name']

    ref_pcd_dir = osp.join(scene, str(collated_dict['ref_frame']))
    src_pcd_dir = osp.join(scene, str(collated_dict['src_frame']))

    image_list_rgb = os.listdir(osp.join('data', 'IndoorLRS', 'image', scene, 'rgb'))
    # image_list_depth = os.listdir(osp.join(self.data_path, 'image', self.collated_dict['scene_name'], 'depth'))
    # sort the image list
    import re
    image_list_rgb.sort(key=lambda y: int(re.findall('\d+', y)[0]))
    image_list_rgb_ref = image_list_rgb[3 * collated_dict['ref_frame']: 3 * (collated_dict['ref_frame'] + 1)]
    image_list_rgb_src = image_list_rgb[3 * collated_dict['src_frame']: 3 * (collated_dict['src_frame'] + 1)]

    # image_list_depth.sort(key=lambda y: int(re.findall('\d+', y)[0]))

    image_pose_path = osp.join('data', 'IndoorLRS', 'image', f'pose_{scene}.log')
    f = open(image_pose_path)
    image_pose = f.readlines()
    f.close()

    image_pose_rgb_ref = image_pose[
                         5 * 3 * collated_dict['ref_frame']: 5 * 3 * (collated_dict['ref_frame'] + 1)]
    image_pose_rgb_src = image_pose[
                         5 * 3 * collated_dict['src_frame']: 5 * 3 * (collated_dict['src_frame'] + 1)]

    # pose of fragment as the fragment pcd of IndoorLRS is not the first frame pcd locally, but globally
    fragment_pose_path = osp.join('data', 'IndoorLRS', 'image', f'pose_slac_{scene}.log')
    f = open(fragment_pose_path)
    fragment_pose = f.readlines()
    f.close()
    fragment_pose_ref = fragment_pose[5 * collated_dict['ref_frame']: 5 * (collated_dict['ref_frame'] + 1)]
    fragment_pose_src = fragment_pose[5 * collated_dict['src_frame']: 5 * (collated_dict['src_frame'] + 1)]

    #####################################
    # input ref_pcd_dir
    # output
    # (pose_frame_1, pose_frame_2, pose_frame_3, color_ref_1, color_ref_2, color_ref_3, camera_intrinsic) = (
    #     three_view_img_preprocessing_indoorlrs(collated_dict, path_dir, image_list_rgb_ref, image_pose))
    #
    # (pose_frame_1, pose_frame_2, pose_frame_3, color_src_1, color_src_2, color_src_3, camera_intrinsic) = (
    #     three_view_img_preprocessing_indoorlrs(collated_dict, path_dir, image_list_rgb_src, image_pose))

    # if aug_src > 0.5:
    #     ref_points = np.matmul(ref_points, aug_rotation.T)
    #     rotation = np.matmul(aug_rotation, rotation)
    #     translation = np.matmul(aug_rotation, translation)
    # else:
    #     src_points = np.matmul(src_points, aug_rotation.T)
    #     rotation = np.matmul(rotation, aug_rotation.T)

    if data_dict['use_augmentation']:
        aug_rot = data_dict['aug_rotation']
        if data_dict['aug_src'] > 0.5:
            ref_1_rot = np.eye(4)
            ref_1_rot[:3, :3] = np.linalg.inv(aug_rot.T)
            ref_1_rot = torch.from_numpy(ref_1_rot).float()

            src_1_rot = torch.eye(4).float()
        else:
            src_1_rot = np.eye(4)
            src_1_rot[:3, :3] = np.linalg.inv(aug_rot.T)
            src_1_rot = torch.from_numpy(src_1_rot).float()

            ref_1_rot = torch.eye(4).float()
    else:
        ref_1_rot = torch.eye(4).float()
        src_1_rot = torch.eye(4).float()

    import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(color_ref_1)
    # plt.show()

    img_patch_all, img_patch_length_all, img_patch_idx_all = [], [], []

    img_patch_length = copy.deepcopy(super_pts_length)
    img_patch_idx = torch.arange(super_pts_coord.shape[0])
    # img_patch_idx_2 = copy.deepcopy(img_patch_idx)
    outlier_idx = []

    # set img_size and minimum img_size
    img_size = 64
    min_size = 10
    # all pts, including ref and src
    for i in range(super_pts_coord.shape[0]):
        # get indices of neighbors of i, removing fake indices
        idx_temp = super_pts_neighbor_index[i, :][super_pts_neighbor_index[i, :] < super_pts_neighbor_coord.shape[0]]
        neigh_i = super_pts_neighbor_coord[idx_temp, :]

        # ref_1_rot = torch.eye(4).float()
        # src_1_rot = torch.eye(4).float()

        # ref
        if i < super_pts_length[0]:
            (pose_frame_1, pose_frame_2, pose_frame_3, color_ref_1, color_ref_2, color_ref_3, camera_intrinsic) = (
                three_view_img_preprocessing_indoorlrs(collated_dict, image_list_rgb_ref, image_pose_rgb_ref))

            # neigh_i_raw = copy.deepcopy(neigh_i)
            # recover the raw pcd before data augmentation
            neigh_i_raw = np.matmul(neigh_i, ref_1_rot[:3, :3])

            pts_to_frame_1, pts_to_frame_2, pts_to_frame_3 = (
                pcd_fragment2frame(fragment_pose_ref, neigh_i_raw,
                                        3, pose_frame_1, pose_frame_2=pose_frame_2, pose_frame_3=pose_frame_3))

            # pts_to_mid = np.matmul(neigh_i_raw, pos_ref_1_rot.T) + pose_ref_1_tra
            # pts_to_frame_1 = copy.deepcopy(neigh_i_raw)
            # pts_to_frame_1 = np.matmul(neigh_i, ref_1_rot[:3, :3])

            # input camera intrinsics, a list of neigh_i pts, RGB image
            # output the width and height of each patch: (i, 64, 64, 3)
            width_1, height_1 = compute_img_range(camera_intrinsic, pts_to_frame_1, color_ref_1)
            img_size_1 = [width_1[1] - width_1[0], height_1[1] - height_1[0]]

            # pts_to_frame_2_tra = pts_to_mid - pose_ref_2_tra
            # pts_to_frame_2 = np.matmul(pts_to_frame_2_tra, pos_ref_2_rot)
            width_2, height_2 = compute_img_range(camera_intrinsic, pts_to_frame_2, color_ref_2)
            img_size_2 = [width_2[1] - width_2[0], height_2[1] - height_2[0]]

            # pts_to_frame_3_tra = pts_to_mid - pose_ref_3_tra
            # pts_to_frame_3 = np.matmul(pts_to_frame_3_tra, pos_ref_3_rot)
            width_3, height_3 = compute_img_range(camera_intrinsic, pts_to_frame_3, color_ref_3)
            img_size_3 = [width_3[1] - width_3[0], height_3[1] - height_3[0]]

            # mask_1 = (img_size_1[0] >= img_size_2[0]) & (img_size_1[1] >= img_size_2[1])
            # mask_2 = (img_size_1[0] >= img_size_3[0]) & (img_size_1[1] >= img_size_3[1])
            # mask_3 = (img_size_2[0] >= img_size_3[0]) & (img_size_2[1] >= img_size_3[1])

            mask_1_ = np.sum(img_size_1) >= np.sum(img_size_2)
            mask_2_ = np.sum(img_size_1) >= np.sum(img_size_3)
            mask_3_ = np.sum(img_size_2) >= np.sum(img_size_3)

            # print(ref_pcd_dir, i)
            img_patch = []

            # if ref_pcd_dir != 'train/sun3d-mit_76_417-76-417b_4/cloud_bin_163.pth' or i != 82:
            # if ref_pcd_dir != 'train/sun3d-harvard_c3-hv_c3_1/cloud_bin_23.pth' or i != 44:
            #     continue

            # mask_1 = (width_1[1] - width_1[0] > img_size) & (height_1[1] - height_1[0] > img_size)
            if all(num >= min_size for num in img_size_1) and mask_1_ and mask_2_:
                img_patch = extract_img_patch(color_ref_1, width_1, height_1, img_size=img_size)
                # img_patch_2 = extract_img_patch(color_ref_2, width_2, height_2, img_size=64)
                # img_patch_3 = extract_img_patch(color_ref_3, width_3, height_3, img_size=64)
            elif all(num >= min_size for num in img_size_2) and not mask_1_ and mask_3_:
                # img_patch_1 = extract_img_patch(color_ref_1, width_1, height_1, img_size=64)
                img_patch = extract_img_patch(color_ref_2, width_2, height_2, img_size=img_size)
                # img_patch_3 = extract_img_patch(color_ref_3, width_3, height_3, img_size=64)
            elif all(num >= min_size for num in img_size_3):
                # img_patch_1 = extract_img_patch(color_ref_1, width_1, height_1, img_size=64)
                # img_patch_2 = extract_img_patch(color_ref_2, width_2, height_2, img_size=64)
                img_patch = extract_img_patch(color_ref_3, width_3, height_3, img_size=img_size)
            else:
                img_patch_length[0] -= 1
                # remove certain index
                outlier_idx.append(i)
                # img_patch_idx_2 = torch.cat((img_patch_idx[:i], img_patch_idx[i + 1:]))

            # plt.figure()
            # plt.imshow(img_patch)
            # plt.show()
        # src
        else:
            (pose_frame_1, pose_frame_2, pose_frame_3, color_src_1, color_src_2, color_src_3, camera_intrinsic) = (
                three_view_img_preprocessing_indoorlrs(collated_dict, image_list_rgb_src, image_pose_rgb_src))

            neigh_i_raw = np.matmul(neigh_i, src_1_rot[:3, :3])

            pts_to_frame_1, pts_to_frame_2, pts_to_frame_3 = (
                pcd_fragment2frame(fragment_pose_src, neigh_i_raw,
                                        3, pose_frame_1, pose_frame_2=pose_frame_2, pose_frame_3=pose_frame_3))

            # pts_to_mid = np.matmul(neigh_i_raw, pos_src_1_rot.T) + pose_src_1_tra
            # pts_to_frame_1 = np.matmul(neigh_i, src_1_rot[:3, :3])
            # pts_to_frame_1 = copy.deepcopy(neigh_i)

            # input camera intrinsics, a list of neigh_i pts, RGB image
            # output the width and height of each patch: (i, 64, 64, 3)
            width_1, height_1 = compute_img_range(camera_intrinsic, pts_to_frame_1, color_src_1)
            img_size_1 = [width_1[1] - width_1[0], height_1[1] - height_1[0]]

            # pts_to_frame_2_tra = pts_to_mid - pose_src_2_tra
            # pts_to_frame_2 = np.matmul(pts_to_frame_2_tra, pos_src_2_rot)
            width_2, height_2 = compute_img_range(camera_intrinsic, pts_to_frame_2, color_src_2)
            img_size_2 = [width_2[1] - width_2[0], height_2[1] - height_2[0]]

            # pts_to_frame_3_tra = pts_to_mid - pose_src_3_tra
            # pts_to_frame_3 = np.matmul(pts_to_frame_3_tra, pos_src_3_rot)
            width_3, height_3 = compute_img_range(camera_intrinsic, pts_to_frame_3, color_src_3)
            img_size_3 = [width_3[1] - width_3[0], height_3[1] - height_3[0]]

            mask_1_ = np.sum(img_size_1) >= np.sum(img_size_2)
            mask_2_ = np.sum(img_size_1) >= np.sum(img_size_3)
            mask_3_ = np.sum(img_size_2) >= np.sum(img_size_3)

            # print(src_pcd_dir, i)
            img_patch = []

            # mask_1 = (width_1[1] - width_1[0] > img_size) & (height_1[1] - height_1[0] > img_size)
            if all(num >= min_size for num in img_size_1) and mask_1_ and mask_2_:
                img_patch = extract_img_patch(color_src_1, width_1, height_1, img_size=img_size)
                # img_patch_2 = extract_img_patch(color_ref_2, width_2, height_2, img_size=64)
                # img_patch_3 = extract_img_patch(color_ref_3, width_3, height_3, img_size=64)
            elif all(num >= min_size for num in img_size_2) and not mask_1_ and mask_3_:
                # img_patch_1 = extract_img_patch(color_ref_1, width_1, height_1, img_size=64)
                img_patch = extract_img_patch(color_src_2, width_2, height_2, img_size=img_size)
                # img_patch_3 = extract_img_patch(color_ref_3, width_3, height_3, img_size=64)
            elif all(num >= min_size for num in img_size_3):
                # img_patch_1 = extract_img_patch(color_ref_1, width_1, height_1, img_size=64)
                # img_patch_2 = extract_img_patch(color_ref_2, width_2, height_2, img_size=64)
                img_patch = extract_img_patch(color_src_3, width_3, height_3, img_size=img_size)
            else:
                img_patch_length[1] -= 1
                # remove certain index
                # img_patch_idx_2 = torch.cat((img_patch_idx[:i], img_patch_idx[i + 1:]))
                outlier_idx.append(i)

        # print(f'neighs: {neigh_i.shape}')
        if img_patch != []:
            img_patch_all.append(img_patch)
        # img_patch_length_all.append(img_patch_length)
        # img_patch_idx_all.append(img_patch_idx)
    collated_dict['img_patch'] = img_patch_all
    collated_dict['img_patch_length'] = img_patch_length

    img_patch_idx = np.delete(img_patch_idx, outlier_idx)
    collated_dict['img_patch_idx'] = img_patch_idx
    ###################################################

    return collated_dict


def registration_collate_fn_stack_mode_scannetpp(
    data_dicts, num_stages, voxel_size, search_radius, neighbor_limits, precompute_data=True
):
    r"""Collate function for registration in stack mode.

    Points are organized in the following order: [ref_1, ..., ref_B, src_1, ..., src_B].
    The correspondence indices are within each point cloud without accumulation.

    Args:
        data_dicts (List[Dict])
        num_stages (int)
        voxel_size (float)
        search_radius (float)
        neighbor_limits (List[int])
        precompute_data (bool)

    Returns:
        collated_dict (Dict)
    """
    batch_size = len(data_dicts)
    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    # handle special keys: [ref_feats, src_feats] -> feats, [ref_points, src_points] -> points, lengths
    feats = torch.cat(collated_dict.pop('ref_feats') + collated_dict.pop('src_feats'), dim=0)
    points_list = collated_dict.pop('ref_points') + collated_dict.pop('src_points')
    lengths = torch.LongTensor([points.shape[0] for points in points_list])
    points = torch.cat(points_list, dim=0)

    if batch_size == 1:
        # remove wrapping brackets if batch_size is 1
        for key, value in collated_dict.items():
            collated_dict[key] = value[0]

    collated_dict['features'] = feats
    if precompute_data:
        input_dict = precompute_data_stack_mode(points, lengths, num_stages, voxel_size, search_radius, neighbor_limits)
        collated_dict.update(input_dict)
    else:
        collated_dict['points'] = points
        collated_dict['lengths'] = lengths
    collated_dict['batch_size'] = batch_size

    ############################################3
    # add img patch
    ##########################################################
    # img patch extraction
    super_pts_coord = collated_dict['points'][-1]
    super_pts_length = collated_dict['lengths'][-1]
    super_pts_neighbor_coord = collated_dict['points'][-2]
    super_pts_neighbor_index = collated_dict['subsampling'][-1]

    ref_pcd_dir = collated_dict['metadata']['pcd0']
    src_pcd_dir = collated_dict['metadata']['pcd1']

    #####################################
    # input ref_pcd_dir
    # output
    (pos_ref_1_rot, pose_ref_1_tra, pos_ref_2_rot, pose_ref_2_tra, pos_ref_3_rot, pose_ref_3_tra,
     color_ref_1, color_ref_2, color_ref_3, camera_intrinsic_1, camera_intrinsic_2, camera_intrinsic_3) = (
        three_view_img_preprocessing_scannetpp(collated_dict, ref_pcd_dir))

    (pos_src_1_rot, pose_src_1_tra, pos_src_2_rot, pose_src_2_tra, pos_src_3_rot, pose_src_3_tra,
     color_src_1, color_src_2, color_src_3, camera_intrinsic_4, camera_intrinsic_5, camera_intrinsic_6) = (
        three_view_img_preprocessing_scannetpp(collated_dict, src_pcd_dir))

    image_rgb_size = [1440, 1920]
    image_depth_size = [192, 256]
    image_depth_size_2 = [480, 640]

    # resize img and intrinsics
    camera_intrinsic_1 = get_depth_intrinsics(camera_intrinsic_1, image_rgb_size, image_depth_size_2)
    camera_intrinsic_2 = get_depth_intrinsics(camera_intrinsic_2, image_rgb_size, image_depth_size_2)
    camera_intrinsic_3 = get_depth_intrinsics(camera_intrinsic_3, image_rgb_size, image_depth_size_2)
    camera_intrinsic_4 = get_depth_intrinsics(camera_intrinsic_4, image_rgb_size, image_depth_size_2)
    camera_intrinsic_5 = get_depth_intrinsics(camera_intrinsic_5, image_rgb_size, image_depth_size_2)
    camera_intrinsic_6 = get_depth_intrinsics(camera_intrinsic_6, image_rgb_size, image_depth_size_2)
    color_ref_1 = cv2.resize(color_ref_1, (640, 480), interpolation=cv2.INTER_AREA)
    color_ref_2 = cv2.resize(color_ref_2, (640, 480), interpolation=cv2.INTER_AREA)
    color_ref_3 = cv2.resize(color_ref_3, (640, 480), interpolation=cv2.INTER_AREA)
    color_src_1 = cv2.resize(color_src_1, (640, 480), interpolation=cv2.INTER_AREA)
    color_src_2 = cv2.resize(color_src_2, (640, 480), interpolation=cv2.INTER_AREA)
    color_src_3 = cv2.resize(color_src_3, (640, 480), interpolation=cv2.INTER_AREA)


    # if aug_src > 0.5:
    #     ref_points = np.matmul(ref_points, aug_rotation.T)
    #     rotation = np.matmul(aug_rotation, rotation)
    #     translation = np.matmul(aug_rotation, translation)
    # else:
    #     src_points = np.matmul(src_points, aug_rotation.T)
    #     rotation = np.matmul(rotation, aug_rotation.T)

    if data_dict['use_augmentation']:
        aug_rot = data_dict['aug_rotation']
        if data_dict['aug_src'] > 0.5:
            ref_1_rot = np.eye(4)
            ref_1_rot[:3, :3] = np.linalg.inv(aug_rot.T)
            ref_1_rot = torch.from_numpy(ref_1_rot).float()

            src_1_rot = torch.eye(4).float()
        else:
            src_1_rot = np.eye(4)
            src_1_rot[:3, :3] = np.linalg.inv(aug_rot.T)
            src_1_rot = torch.from_numpy(src_1_rot).float()

            ref_1_rot = torch.eye(4).float()
    else:
        ref_1_rot = torch.eye(4).float()
        src_1_rot = torch.eye(4).float()

    import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(color_ref_1)
    # plt.show()

    img_patch_all, img_patch_length_all, img_patch_idx_all = [], [], []

    img_patch_length = copy.deepcopy(super_pts_length)
    img_patch_idx = torch.arange(super_pts_coord.shape[0])
    # img_patch_idx_2 = copy.deepcopy(img_patch_idx)
    outlier_idx = []

    # set img_size and minimum img_size
    img_size = 64
    min_size = 10
    # all pts, including ref and src
    for i in range(super_pts_coord.shape[0]):
        # get indices of neighbors of i, removing fake indices
        idx_temp = super_pts_neighbor_index[i, :][super_pts_neighbor_index[i, :] < super_pts_neighbor_coord.shape[0]]
        neigh_i = super_pts_neighbor_coord[idx_temp, :]

        # ref_1_rot = torch.eye(4).float()
        # src_1_rot = torch.eye(4).float()

        # ref
        if i < super_pts_length[0]:
            # neigh_i_raw = copy.deepcopy(neigh_i)
            # recover the raw pcd before data augmentation
            neigh_i_raw = np.matmul(neigh_i, ref_1_rot[:3, :3])
            pts_to_mid = np.matmul(neigh_i_raw, pos_ref_1_rot.T) + pose_ref_1_tra
            pts_to_frame_1 = copy.deepcopy(neigh_i_raw)
            # pts_to_frame_1 = np.matmul(neigh_i, ref_1_rot[:3, :3])

            # input camera intrinsics, a list of neigh_i pts, RGB image
            # output the width and height of each patch: (i, 64, 64, 3)
            width_1, height_1 = compute_img_range(camera_intrinsic_1, pts_to_frame_1, color_ref_1)
            img_size_1 = [width_1[1] - width_1[0], height_1[1] - height_1[0]]

            pts_to_frame_2_tra = pts_to_mid - pose_ref_2_tra
            pts_to_frame_2 = np.matmul(pts_to_frame_2_tra, pos_ref_2_rot)
            width_2, height_2 = compute_img_range(camera_intrinsic_2, pts_to_frame_2, color_ref_2)
            img_size_2 = [width_2[1] - width_2[0], height_2[1] - height_2[0]]

            pts_to_frame_3_tra = pts_to_mid - pose_ref_3_tra
            pts_to_frame_3 = np.matmul(pts_to_frame_3_tra, pos_ref_3_rot)
            width_3, height_3 = compute_img_range(camera_intrinsic_3, pts_to_frame_3, color_ref_3)
            img_size_3 = [width_3[1] - width_3[0], height_3[1] - height_3[0]]

            # mask_1 = (img_size_1[0] >= img_size_2[0]) & (img_size_1[1] >= img_size_2[1])
            # mask_2 = (img_size_1[0] >= img_size_3[0]) & (img_size_1[1] >= img_size_3[1])
            # mask_3 = (img_size_2[0] >= img_size_3[0]) & (img_size_2[1] >= img_size_3[1])

            mask_1_ = np.sum(img_size_1) >= np.sum(img_size_2)
            mask_2_ = np.sum(img_size_1) >= np.sum(img_size_3)
            mask_3_ = np.sum(img_size_2) >= np.sum(img_size_3)

            # print(ref_pcd_dir, i)
            img_patch = []

            # if ref_pcd_dir != 'train/sun3d-mit_76_417-76-417b_4/cloud_bin_163.pth' or i != 82:
            # if ref_pcd_dir != 'train/sun3d-harvard_c3-hv_c3_1/cloud_bin_23.pth' or i != 44:
            #     continue

            # mask_1 = (width_1[1] - width_1[0] > img_size) & (height_1[1] - height_1[0] > img_size)
            if all(num >= min_size for num in img_size_1) and mask_1_ and mask_2_:
                img_patch = extract_img_patch(color_ref_1, width_1, height_1, img_size=img_size)
                # img_patch_2 = extract_img_patch(color_ref_2, width_2, height_2, img_size=64)
                # img_patch_3 = extract_img_patch(color_ref_3, width_3, height_3, img_size=64)
            elif all(num >= min_size for num in img_size_2) and not mask_1_ and mask_3_:
                # img_patch_1 = extract_img_patch(color_ref_1, width_1, height_1, img_size=64)
                img_patch = extract_img_patch(color_ref_2, width_2, height_2, img_size=img_size)
                # img_patch_3 = extract_img_patch(color_ref_3, width_3, height_3, img_size=64)
            elif all(num >= min_size for num in img_size_3):
                # img_patch_1 = extract_img_patch(color_ref_1, width_1, height_1, img_size=64)
                # img_patch_2 = extract_img_patch(color_ref_2, width_2, height_2, img_size=64)
                img_patch = extract_img_patch(color_ref_3, width_3, height_3, img_size=img_size)
            else:
                img_patch_length[0] -= 1
                # remove certain index
                outlier_idx.append(i)
                # img_patch_idx_2 = torch.cat((img_patch_idx[:i], img_patch_idx[i + 1:]))

            # plt.figure()
            # plt.imshow(img_patch)
            # plt.show()
        # src
        else:
            neigh_i_raw = np.matmul(neigh_i, src_1_rot[:3, :3])

            pts_to_mid = np.matmul(neigh_i_raw, pos_src_1_rot.T) + pose_src_1_tra
            pts_to_frame_1 = np.matmul(neigh_i, src_1_rot[:3, :3])
            # pts_to_frame_1 = copy.deepcopy(neigh_i)

            # input camera intrinsics, a list of neigh_i pts, RGB image
            # output the width and height of each patch: (i, 64, 64, 3)
            width_1, height_1 = compute_img_range(camera_intrinsic_4, pts_to_frame_1, color_src_1)
            img_size_1 = [width_1[1] - width_1[0], height_1[1] - height_1[0]]

            pts_to_frame_2_tra = pts_to_mid - pose_src_2_tra
            pts_to_frame_2 = np.matmul(pts_to_frame_2_tra, pos_src_2_rot)
            width_2, height_2 = compute_img_range(camera_intrinsic_5, pts_to_frame_2, color_src_2)
            img_size_2 = [width_2[1] - width_2[0], height_2[1] - height_2[0]]

            pts_to_frame_3_tra = pts_to_mid - pose_src_3_tra
            pts_to_frame_3 = np.matmul(pts_to_frame_3_tra, pos_src_3_rot)
            width_3, height_3 = compute_img_range(camera_intrinsic_6, pts_to_frame_3, color_src_3)
            img_size_3 = [width_3[1] - width_3[0], height_3[1] - height_3[0]]

            mask_1_ = np.sum(img_size_1) >= np.sum(img_size_2)
            mask_2_ = np.sum(img_size_1) >= np.sum(img_size_3)
            mask_3_ = np.sum(img_size_2) >= np.sum(img_size_3)

            # print(src_pcd_dir, i)
            img_patch = []

            # mask_1 = (width_1[1] - width_1[0] > img_size) & (height_1[1] - height_1[0] > img_size)
            if all(num >= min_size for num in img_size_1) and mask_1_ and mask_2_:
                img_patch = extract_img_patch(color_src_1, width_1, height_1, img_size=img_size)
                # img_patch_2 = extract_img_patch(color_ref_2, width_2, height_2, img_size=64)
                # img_patch_3 = extract_img_patch(color_ref_3, width_3, height_3, img_size=64)
            elif all(num >= min_size for num in img_size_2) and not mask_1_ and mask_3_:
                # img_patch_1 = extract_img_patch(color_ref_1, width_1, height_1, img_size=64)
                img_patch = extract_img_patch(color_src_2, width_2, height_2, img_size=img_size)
                # img_patch_3 = extract_img_patch(color_ref_3, width_3, height_3, img_size=64)
            elif all(num >= min_size for num in img_size_3):
                # img_patch_1 = extract_img_patch(color_ref_1, width_1, height_1, img_size=64)
                # img_patch_2 = extract_img_patch(color_ref_2, width_2, height_2, img_size=64)
                img_patch = extract_img_patch(color_src_3, width_3, height_3, img_size=img_size)
            else:
                img_patch_length[1] -= 1
                # remove certain index
                # img_patch_idx_2 = torch.cat((img_patch_idx[:i], img_patch_idx[i + 1:]))
                outlier_idx.append(i)

        # print(f'neighs: {neigh_i.shape}')
        if img_patch != []:
            img_patch_all.append(img_patch)
        # img_patch_length_all.append(img_patch_length)
        # img_patch_idx_all.append(img_patch_idx)
    collated_dict['img_patch'] = img_patch_all
    collated_dict['img_patch_length'] = img_patch_length

    img_patch_idx = np.delete(img_patch_idx, outlier_idx)
    collated_dict['img_patch_idx'] = img_patch_idx
    ###################################################

    return collated_dict


def calibrate_neighbors_stack_mode(
    dataset, collate_fn, num_stages, voxel_size, search_radius, keep_ratio=0.8, sample_threshold=2000
):
    # Compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (search_radius / voxel_size + 1) ** 3))
    neighbor_hists = np.zeros((num_stages, hist_n), dtype=np.int32)
    max_neighbor_limits = [hist_n] * num_stages

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        data_dict = collate_fn(
            [dataset[i]], num_stages, voxel_size, search_radius, max_neighbor_limits, precompute_data=True
        )

        # update histogram
        counts = [np.sum(neighbors.numpy() < neighbors.shape[0], axis=1) for neighbors in data_dict['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighbor_hists += np.vstack(hists)

        if np.min(np.sum(neighbor_hists, axis=1)) > sample_threshold:
            break

    cum_sum = np.cumsum(neighbor_hists.T, axis=0)
    neighbor_limits = np.sum(cum_sum < (keep_ratio * cum_sum[hist_n - 1, :]), axis=0)

    return neighbor_limits


def build_dataloader_stack_mode(
    dataset,
    collate_fn,
    num_stages,
    voxel_size,
    search_radius,
    neighbor_limits,
    batch_size=1,
    num_workers=1,
    shuffle=False,
    drop_last=False,
    distributed=False,
    precompute_data=True,
):
    dataloader = build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=partial(
            collate_fn,
            num_stages=num_stages,
            voxel_size=voxel_size,
            search_radius=search_radius,
            neighbor_limits=neighbor_limits,
            precompute_data=precompute_data,
        ),
        drop_last=drop_last,
        distributed=distributed,
    )
    return dataloader
