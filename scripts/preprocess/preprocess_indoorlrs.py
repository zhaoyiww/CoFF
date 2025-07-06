import os
import os.path as osp
import pickle
import numpy as np
import open3d as o3d
import shutil
# import cv2
import time
from PIL import Image
import torch
import copy
import argparse
import pandas as pd
from scipy.spatial import cKDTree
import re
from easydict import EasyDict as edict
from tqdm import tqdm
from utils.o3d_tools import get_nearest_neighbor


# source: geotransformer
# def get_nearest_neighbor(src_pts, tgt_pts, return_index=False):
#     s_tree = cKDTree(src_pts)
#     dist, indices = s_tree.query(tgt_pts, k=1, workers=-1)
#     if return_index:
#         return dist, indices
#     else:
#         return dist


# source: geotransformer
def cal_overlap(src_pcd, tgt_pcd, trans_matrix=None, overlap_radius=0.1):
    r"""Compute the overlap of two point clouds."""
    src_pcd_temp = copy.deepcopy(src_pcd)
    tgt_pcd_temp = copy.deepcopy(tgt_pcd)
    if trans_matrix is not None:
        # src_points = array_transform(src_pcd, transform)
        src_pcd = src_pcd_temp.transform(trans_matrix)
    # src_pcd.paint_uniform_color([0, 0.651, 0.929])
    # o3d.visualization.draw_geometries([src_pcd, tgt_pcd_temp])
    src_pts, tgt_pts = src_pcd.points, tgt_pcd_temp.points
    nn_distances = get_nearest_neighbor(src_pts, tgt_pts)
    overlap = np.mean(nn_distances < overlap_radius)
    return overlap


# do transformation for ndarray format point cloud
def array_transform(src_pts, est_transform):
    rotation, translation = est_transform[:3, :3], est_transform[:3, 3]
    src_pts = np.matmul(src_pts, rotation.T) + translation
    return src_pts


class IndoorRGBD_Dataset(object):
    """
    cal_pkl
    cal_npz
    """

    def __init__(self, config):
        self.root = config.root
        # self.voxel_size = config.voxel_size
        # self.neighbor_radius = config.neighbor_radius
        # self.img_size = config.img_size
        self.dataset_type = config.dataset_type
        self.overlap_radius = config.overlap_radius
        # self.camera_intrinsic = np.loadtxt(osp.join(self.root, "camera-intrinsics.txt"))

        with open(osp.join(self.root, f"{self.dataset_type}_list.txt")) as f:
            scene_list = f.readlines()
            scene_list = [scene.strip("\n") for scene in scene_list]

        self.scene_list = scene_list
        self.get_pkl()
        self.get_pcd_npz()

    def get_pkl(self):

        pkl_path = osp.join("./configs/IndoorRGBD")
        os.makedirs(pkl_path, exist_ok=True)

        info = dict()
        rot, trans, src, tgt, overlap = [], [], [], [], []
        for scene in self.scene_list:
            gt_transform_path = osp.join(self.root, f"fragments/{scene}/reg_output.log")

            # each column: tgt_id, src_id, xx
            pairs_id = pd.read_csv(gt_transform_path, skiprows=lambda x: x % 5 != 0, sep="\s+", na_values="-1",
                                   header=None)
            pairs_id = pairs_id.to_numpy()

            gt_transform = pd.read_csv(gt_transform_path, skiprows=lambda x: x % 5 == 0, sep="\s+", na_values="-1",
                                       header=None)
            gt_transform = gt_transform.to_numpy()

            for pair in tqdm(range(len(pairs_id))):
                tgt_temp = osp.join(self.dataset_type, scene, f"mesh_{pairs_id[pair][0]}.npz")
                src_temp = osp.join(self.dataset_type, scene, f"mesh_{pairs_id[pair][1]}.npz")
                tgt.append(tgt_temp)
                src.append(src_temp)

                # tgt = transform * src
                rot.append(gt_transform[pair * 4: pair * 4 + 3, :3])
                trans_temp = gt_transform[pair * 4: pair * 4 + 3, 3]
                trans.append(trans_temp[:, None])

                src_pcd_path = osp.join(self.root,
                                            src_temp.replace(self.dataset_type, "fragments").replace('.npz', '.ply'))
                tgt_pcd_path = osp.join(self.root,
                                            tgt_temp.replace(self.dataset_type, "fragments").replace('.npz', '.ply'))

                src_pcd = o3d.io.read_point_cloud(src_pcd_path)
                tgt_pcd = o3d.io.read_point_cloud(tgt_pcd_path)
                # overlap will be different? depends on which pcd is the base one
                overlap_temp = cal_overlap(src_pcd, tgt_pcd, gt_transform[pair * 4: pair * 4 + 4], self.overlap_radius)
                overlap.append(overlap_temp)

            print(f"'.pkl' data of scene: {scene} is ready")
            time.sleep(0.05)

        info["src"] = src
        info["tgt"] = tgt
        info["rot"] = rot
        info["trans"] = trans
        info["overlap"] = overlap

        with open(osp.join(pkl_path, f"{self.dataset_type}_converted.pkl"), "wb") as f:
            pickle.dump(info, f)

        # copy the "xxx_list.txt"" to the destination dir
        shutil.copy2(osp.join(self.root, f"{self.dataset_type}_list.txt"), pkl_path)
        print(f"\n'{osp.join(pkl_path, f'{self.dataset_type}_info.pkl')}' is saved\n")
        time.sleep(0.05)

    def get_pcd_npz(self):
        for scene in self.scene_list:
            npz_path = osp.join("./data/IndoorRGBD", self.dataset_type, scene)
            os.makedirs(npz_path, exist_ok=True)

            # img_path = osp.join(self.root, "images")
            # img_pose = osp.join(img_path, scene + "_pose.log")
            # img_pose = pd.read_csv(img_pose, skiprows=lambda x: x % 5 == 0, sep="\s+", na_values="-1",
            #                           header=None)

            scene_path = osp.join(self.root, "fragments", scene)
            # frags_pose = osp.join(scene_path, "pose_slac.log")
            # frags_pose = pd.read_csv(frags_pose, skiprows=lambda x: x % 5 == 0, sep="\s+", na_values="-1", header=None)

            pcd_list = os.listdir(scene_path)
            pcd_list = [x for x in pcd_list if x.endswith('.ply')]
            pcd_list.sort(key=lambda y: int(re.findall('\d+', y)[0]))
            assert len(pcd_list) > 0, f"Could not find any '.ply' point cloud under {scene_path}\n"

            # img_list = os.listdir(osp.join(img_path, scene))
            # for i, pcd_path in enumerate(pcd_list[:]):
            for i in tqdm(range(len(pcd_list))):
                if not pcd_list[i].endswith(".ply"):
                    continue
                full_path = osp.join(scene_path, pcd_list[i])
                pcd = o3d.io.read_point_cloud(full_path)

                pts = np.asarray(pcd.points)
                sample_pts = pts
                sample_clr = np.asarray(pcd.colors)

                # save the raw format data, not preprocessed data
                np.savez_compressed(
                    osp.join(npz_path, pcd_list[i].replace(".ply", ".npz")),
                    pcd=np.hstack((np.array(pcd.points), np.array(pcd.colors))),
                    xyz=sample_pts,
                    rgb=sample_clr)

            print(f"'{npz_path}' is saved\n")
            time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser("train the 2D-3D jointing learning models")
    parser.add_argument("--input_root", default="./IndoorRGBD/", type=str, help="path to the dataset")
    parser.add_argument("--output_root", default="./IndoorRGBD/", type=str, help="path to the dataset")
    parser.add_argument("--dataset_type", default="test", type=str, help="")
    parser.add_argument("--overlap_radius", default=0.025, type=float, help="")
    args = parser.parse_args()

    config = edict(vars(args))

    shutil.copy2('generate_pkl_pcd.py', config.root)

    IndoorRGBD_Dataset(config)


if __name__ == '__main__':
    main()