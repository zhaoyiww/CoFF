# ---------------------------------------------------------------------------- #
# source from: https://github.com/XuyangBai/PPF-FoldNet/blob/master/script/fuse_fragments_3DMatch.py
# Fuse rgbd frames into fragments in ScanNet++ using TSDF fusion
# - Use existing camera poses
# - Save colors & normals
# ---------------------------------------------------------------------------- #
from pathlib import Path
import argparse
import math
import numpy as np
import os.path as osp
import sys
import json
import copy
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import open3d as o3d
from scripts.preprocess.utils import io2 as uio


def read_rgbd_image(cfg, color_file, depth_file, convert_rgb_to_intensity):
    """ Read separate RGB and depth frames, and convert them into an Open3D RGBD image """
    if color_file is None:
        color_file = depth_file  # to avoid "Unsupported image format."
    color = o3d.io.read_image(color_file)
    color_np = np.asarray(color)
    rescaled_color = cv2.resize(color_np, (cfg.depth_width, cfg.depth_height), interpolation=cv2.INTER_AREA)
    color_rescaled = o3d.geometry.Image(rescaled_color)

    depth = o3d.io.read_image(depth_file)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_rescaled, depth, cfg.depth_scale, cfg.depth_trunc, convert_rgb_to_intensity)

    # visualize rgb and depth
    # plt.subplot(1, 2, 1)
    # plt.title('Redwood grayscale image')
    # plt.imshow(rgbd_image.color)
    # plt.subplot(1, 2, 2)
    # plt.title('Redwood depth image')
    # plt.imshow(rgbd_image.depth)
    # plt.show()
    return rgbd_image


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


def process_single_fragment(cfg, color_files, depth_files, frag_id, n_frags, out_folder, scene):
    depth_only_flag = (len(color_files) == 0)
    n_frames = len(depth_files)

    if depth_only_flag:
        color_type = o3d.pipelines.integration.TSDFVolumeColorType.__dict__['None']
    else:
        color_type = o3d.pipelines.integration.TSDFVolumeColorType.__dict__['RGB8']

    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=cfg.tsdf_cubic_size / 512.0,
                                                          sdf_trunc=0.04,
                                                          color_type=color_type)

    sid = frag_id * cfg.frames_per_frag
    eid = min(sid + cfg.frames_per_frag, n_frames)
    pose_base2world = None
    pose_base2world_inv = None
    for fid in range(sid, eid):
        if not depth_only_flag:
            color_path = color_files[fid]
        else:
            color_path = None

        m = cfg.intrinsic_matrices[fid]
        # get depth intrinsic
        m = get_depth_intrinsics(m, [cfg.rgb_height, cfg.rgb_width],
                                 [cfg.depth_height, cfg.depth_width])
        intrinsic = o3d.camera.PinholeCameraIntrinsic(cfg.depth_width, cfg.depth_height,
                                                      m[0, 0], m[1, 1], m[0, 2], m[1, 2])
        depth_path = depth_files[fid]
        pose_cam2world = cfg.pose_matrices[fid]
        if pose_cam2world is None:
            continue
        if fid == sid:  # Use as base frame
            pose_base2world = pose_cam2world
            pose_base2world_inv = np.linalg.inv(pose_base2world)
        if pose_base2world_inv is None:
            break
        # Relative camera pose
        pose_cam2world = np.matmul(pose_base2world_inv, pose_cam2world)

        rgbd = read_rgbd_image(cfg, color_path, depth_path, False)
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose_cam2world))
    if pose_base2world_inv is None:
        return

    pcloud = volume.extract_point_cloud()
    pcloud.estimate_normals()
    o3d.io.write_point_cloud(osp.join(out_folder, 'cloud_bin_{}.ply'.format(frag_id)), pcloud)

    with open(osp.join(out_folder, f'cloud_bin_{frag_id}.pose.txt'),
              'w') as file:
        # write scene_id, scene_id, src_id, tgt_id
        file.write(f"{scene}\t{scene}\t{sid}\t{eid - 1}\n")
        # write gt_transform
        for row in pose_base2world:
            file.write('\t'.join(f"{value:.8f}" for value in row) + '\n')


# ---------------------------------------------------------------------------- #
# Iterate Folders
# ---------------------------------------------------------------------------- #
def run_frame(cfg, scene):
    scene_folder = osp.join(cfg.input_root, 'image', scene)
    color_names = uio.list_files(osp.join(scene_folder, 'rgb'), '*.jpg')
    color_paths = [osp.join(scene_folder, 'rgb', cf) for cf in color_names]
    depth_names = uio.list_files(osp.join(scene_folder, 'depth'), '*.png')
    depth_paths = [osp.join(scene_folder, 'depth', df) for df in depth_names]

    # n_frames = len(color_paths)
    n_frames = len(depth_paths)
    n_frags = int(math.ceil(float(n_frames) / cfg.frames_per_frag))

    out_folder = osp.join(cfg.output_root, 'fragments', cfg.dataset_type, scene)
    uio.may_create_folder(out_folder)

    # not used, because for ScanNet++ each frame has an individual camera intrinsic
    # intrinsic_path = osp.join(cfg.input_root, scene, 'camera-intrinsics.txt')

    if cfg.threads > 1:
        # not used
        from joblib import Parallel, delayed
        import multiprocessing

        Parallel(n_jobs=cfg.threads)(
            delayed(process_single_fragment)(cfg, color_paths, depth_paths, frag_id, n_frags, out_folder, scene)
            for frag_id in range(n_frags))
    else:
        if cfg.num_frag_per_scene > 0:
            n_frags = min(n_frags, cfg.num_frag_per_scene)
        for frag_id in tqdm(range(n_frags)):
            process_single_fragment(cfg, color_paths, depth_paths, frag_id, n_frags, out_folder, scene)
    print("    Finished {}".format(scene))


def extract_pose_cam2world(cfg, scene):
    """ Extract camera2world pose from path """
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


def run_scene(cfg, scene):
    print("  Start scene {} ".format(scene))

    # extract pose information
    cfg.pose_matrices, cfg.intrinsic_matrices = extract_pose_cam2world(cfg, scene)
    run_frame(cfg, scene)

    print("  Finished scene {} ".format(scene))


def run(cfg):
    print("Start making fragments...")

    uio.may_create_folder(osp.join(cfg.output_root, 'fragments'))

    # scenes = uio.list_folders(osp.join(cfg.input_root), sort=False)
    with open(osp.join(cfg.input_root, 'metadata', 'split', f"{cfg.dataset_type}_scannetpp.txt")) as f:
        scenes = f.readlines()
        scenes = [scene.strip("\n") for scene in scenes]
    print("{} scenes".format(len(scenes)))
    for scene in tqdm(scenes):
        # if not scene.startswith('analysis'):
        #    continue
        run_scene(cfg, scene)

    print("Finished making fragments")


# ---------------------------------------------------------------------------- #
# Arguments
# ---------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root',
                        default='/scratch2/zhawang/projects/registration/CoFF_local/scripts/preprocess/inputs/')
    parser.add_argument('--output_root',
                        default='/scratch2/zhawang/projects/registration/CoFF_local/scripts/preprocess/inputs/')
    parser.add_argument('--depth_scale', type=float, default=1000.0)
    parser.add_argument('--depth_trunc', type=float, default=4.0)
    parser.add_argument('--frames_per_frag', type=int, default=20)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--num_frag_per_scene', type=int, default=50,
                        help='set to 0 if generate frags for all frames')
    parser.add_argument("--dataset_type", default="test", type=str)
    parser.add_argument('--tsdf_cubic_size', type=float, default=3.0)
    parser.add_argument('--rgb_height', type=int, default=1440)
    parser.add_argument('--rgb_width', type=int, default=1920)
    parser.add_argument('--depth_height', type=int, default=192)
    parser.add_argument('--depth_width', type=int, default=256)

    return parser.parse_args()


if __name__ == '__main__':
    cfg = parse_args()
    run(cfg)
