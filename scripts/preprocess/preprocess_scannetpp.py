"""
Generate benchmarks pose and .pkl for scannet++
"""
import os
import pickle
import numpy as np
import open3d as o3d
# import shutil
# import cv2
import time
# from PIL import Image
import torch
import copy
import argparse
import re
from easydict import EasyDict as edict
from tqdm import tqdm
import os.path as osp
from natsort import natsorted
from utils.o3d_tools import compute_overlap_ratio


class preprocess_ScanNetpp(object):
    """ Preprocess ScanNetPP dataset, generating benchmarks and .pkl file """

    def __init__(self, config):
        self.input_root = config.input_root
        self.output_root = config.output_root
        self.voxel_size = config.voxel_size
        # self.neighbor_radius = config.neighbor_radius
        # self.img_size = config.img_size
        self.dataset_type = config.dataset_type
        self.benchmark_name = config.benchmark_name
        # self.overlap_radius = config.overlap_radius
        self.overlap_ratio_min = config.overlap_ratio_min
        self.overlap_ratio_max = config.overlap_ratio_max
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # self.camera_intrinsic = np.loadtxt(osp.join(self.root, "camera-intrinsics.txt"))

        with open(osp.join(self.input_root, 'metadata', 'split', f"{self.dataset_type}_scannetpp.txt")) as f:
            scene_list = f.readlines()
            scene_list = [scene.strip("\n") for scene in scene_list]

        self.scene_list = scene_list
        start_time = time.time()
        self.get_gt_and_pkl()
        end_time = time.time()
        print('Preprocessing time: {:.2f} seconds'.format(end_time - start_time))

    def get_gt_and_pkl(self):

        metadata_root = osp.join(self.output_root, 'metadata')
        os.makedirs(metadata_root, exist_ok=True)
        # pkl_path = osp.join(metadata_root, f"{self.dataset_type}.pkl")

        info = dict()
        rot, trans, src, tgt, overlap = [], [], [], [], []

        for scene in tqdm(self.scene_list, desc="Num. of scenes"):
            fragments = os.listdir(osp.join(self.input_root, 'fragments', self.dataset_type, scene))
            ply_files = natsorted([file for file in fragments if file.endswith('.ply')])

            # save gt_transform to gt.log
            gt_path = osp.join(metadata_root, 'benchmarks', self.benchmark_name, scene)
            os.makedirs(gt_path, exist_ok=True)
            # initialize gt.log
            with open(osp.join(gt_path, 'gt.log'), 'w') as file:
                pass

            # generate gt.log for all fragments
            for i, src_file in enumerate(ply_files):
                n_continuous = 0
                src_id = re.findall(r'\d+', src_file)[0]
                src_path = osp.join(self.input_root, 'fragments', self.dataset_type, scene)
                src_pose_path = osp.join(src_path, f'cloud_bin_{src_id}.pose.txt')
                src_pose = np.loadtxt(src_pose_path, skiprows=1)
                src_pose = torch.tensor(src_pose, dtype=torch.float64).to(self.device)
                src_pcd = o3d.io.read_point_cloud(osp.join(src_path, src_file))
                # subsample the pcd for faster processing
                src_pcd = src_pcd.voxel_down_sample(self.voxel_size)

                for j, tgt_file in enumerate(tqdm(ply_files[i+1:], position=0, leave=True)):
                    tgt_id = re.findall(r'\d+', tgt_file)[0]
                    tgt_path = osp.join(self.input_root, 'fragments', self.dataset_type, scene)
                    tgt_pose_path = osp.join(tgt_path, f'cloud_bin_{tgt_id}.pose.txt')
                    tgt_pose = np.loadtxt(tgt_pose_path, skiprows=1)
                    tgt_pose = torch.tensor(tgt_pose, dtype=torch.float64).to(self.device)

                    # R1 * X1 + T1 = R2 * X2 + T2
                    gt_transform = torch.matmul(torch.linalg.inv(tgt_pose), src_pose)

                    tgt_pcd = o3d.io.read_point_cloud(osp.join(tgt_path, tgt_file))
                    tgt_pcd = tgt_pcd.voxel_down_sample(self.voxel_size)

                    # set a strict threshold to generate more challenging pairs as this dataset contains enough scenes
                    overlap_temp = compute_overlap_ratio(src_pcd, tgt_pcd, transform=gt_transform.cpu(),
                                                         positive_radius=0.025)
                    # only select pairs with certain range of overlap ratios
                    if self.overlap_ratio_min <= overlap_temp <= self.overlap_ratio_max:
                        n_continuous = 0
                        src_temp = osp.join(self.dataset_type, scene, f"cloud_bin_{src_id}.npz")
                        tgt_temp = osp.join(self.dataset_type, scene, f"cloud_bin_{tgt_id}.npz")
                        src.append(src_temp)
                        tgt.append(tgt_temp)

                        rot.append(gt_transform[:3, :3].cpu().numpy())
                        trans.append(gt_transform[:3, 3][:, None].cpu().numpy())
                        overlap.append(overlap_temp)

                        # gt_path = osp.join(metadata_root, 'benchmarks', scene)
                        os.makedirs(gt_path, exist_ok=True)
                        with open(osp.join(gt_path, 'gt.log'), 'a') as file:
                            # write src_id, tgt_id, num_frame
                            file.write(f"{src_id}\t{tgt_id}\t{len(ply_files)}\n")
                            # write gt_transform
                            for row in gt_transform:
                                file.write('\t'.join(f"{value:.8f}" for value in row) + '\n')
                    else:
                        n_continuous += 1
                    # if existing 3 invalid overlap ratios, jump out the internal loop
                    # assume the frames are cumulatively collected; stop early
                    if n_continuous >= 3 and int(tgt_id) - int(src_id) > 10:
                        break
            print(f"'.pkl' data of scene: {scene} is ready")
            # time.sleep(0.05)

        # prepare info for .pkl file
        info["src"] = src
        info["tgt"] = tgt
        info["rot"] = rot
        info["trans"] = trans
        info["overlap"] = overlap

        # save predator format .pkl file, for preprocessing and later baseline comparison
        # save_path_predator = osp.join(metadata_root,
        #                              f"ScanNetpp_{str(int(self.overlap_ratio_min*10)).zfill(2)}_"
        #                              f"{str(int(self.overlap_ratio_max*10)).zfill(2)}_preprocess.pkl")
        save_path_predator = osp.join(metadata_root,
                                     f"{self.benchmark_name}_converted.pkl")
        with open(save_path_predator, "wb") as f:
            pickle.dump(info, f)
        f.close()

        # save geotrans format .pkl file, for own method evaluation
        ##########
        pkl_list = []
        pkl_single = dict()
        for i in range(len(info['overlap'])):
            pkl_single['overlap'] = info['overlap'][i]
            pkl_single['pcd0'] = info['tgt'][i]
            pkl_single['pcd1'] = info['src'][i]
            # pkl_single['rotation'] = pkl_input['rot'][i, :, :]
            # pkl_single['translation'] = pkl_input['trans'][i, :].squeeze()
            pkl_single['rotation'] = info['rot'][i]
            pkl_single['translation'] = info['trans'][i].squeeze()
            scene_name = osp.basename(osp.dirname(pkl_single['pcd0']))
            frag_id0 = int(re.findall(r'\d+', osp.basename(pkl_single['pcd0']))[0])
            frag_id1 = int(re.findall(r'\d+', osp.basename(pkl_single['pcd1']))[0])
            pkl_single['scene_name'] = scene_name
            pkl_single['frag_id0'] = frag_id0
            pkl_single['frag_id1'] = frag_id1

            pkl_list.append(copy.deepcopy(pkl_single))

        # save_path = osp.join(metadata_root, f"ScanNetpp_{str(int(self.overlap_ratio_min*10)).zfill(2)}_"
        #                                         f"{str(int(self.overlap_ratio_max*10)).zfill(2)}.pkl")

        save_path = osp.join(metadata_root, f"{self.benchmark_name}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(pkl_list, f)
        f.close()
        ##########
        print(f"{save_path} is saved\n")


def main():
    parser = argparse.ArgumentParser("Generate benchmarks and .pkl for ScanNet++")
    parser.add_argument("--input_root",
                        default="/scratch2/zhawang/projects/registration/CoFF_local/scripts/preprocess/inputs/",
                        type=str, help="path to the dataset")
    parser.add_argument("--output_root",
                        default="/scratch2/zhawang/projects/registration/CoFF_local/scripts/preprocess/inputs/",
                        type=str, help="path to the dataset")
    parser.add_argument("--dataset_type", default="test", type=str, help="")
    parser.add_argument("--benchmark_name", default="ScanNetpp", type=str, help="")
    parser.add_argument("--voxel_size", default=0.025, type=float,
                        help="the threshold for computing overlap ratio")
    # parser.add_argument("--overlap_radius", default=0.1, type=float,
    #                     help="the threshold for computing overlap ratio")
    parser.add_argument("--overlap_ratio_min", default=0.10, type=float,
                        help="min. overlap ratio")
    parser.add_argument("--overlap_ratio_max", default=1.0, type=float,
                        help="max. overlap ratio")
    args = parser.parse_args()

    config = edict(vars(args))

    preprocess_ScanNetpp(config)


if __name__ == '__main__':
    main()
