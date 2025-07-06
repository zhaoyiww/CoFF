# Create a planar subset of the ScanNet++ dataset for testing

import pickle
import copy
import os
import numpy as np
import re
import argparse
from scipy.spatial import cKDTree
import open3d as o3d
from utils.common import load_yaml, run_time
from easydict import EasyDict as edict
import os.path as osp
from tqdm import tqdm
from utils.o3d_tools import array2pcd, pcd2array, get_nearest_neighbor
import pyransac3d as pyrsc

def create_scannetpp_planar(config):
    # Load metadata
    with open(osp.join(config.input_root, 'metadata', f'{config.benchmark_name}_converted.pkl'), "rb") as f:
        data = pickle.load(f)

    data["trans"] = np.array(data["trans"])
    data["rot"] = np.array(data["rot"])
    data["overlap"] = np.array(data["overlap"])

    # Filter pairs by overlap
    mask = data['overlap'] <= 1
    lo_set = np.where(mask)
    src = np.array(data['src'])[lo_set]
    tgt = np.array(data['tgt'])[lo_set]
    trans = np.array(data["trans"])[lo_set]
    rot = np.array(data["rot"])[lo_set]
    overlap = data['overlap'][lo_set]

    # Read scene list
    scene_list_dir = os.path.join(config.input_root, 'metadata', 'split', 'test_scannetpp.txt')
    with open(scene_list_dir) as f:
        scenes = [line.rstrip('\n') for line in f.readlines()]

    dataset_name = f'{config.benchmark_name}_planar'
    benchmark = os.path.join(config.output_root, 'metadata', 'benchmarks', dataset_name)
    os.makedirs(benchmark, exist_ok=True)

    src_geo_smooth, tgt_geo_smooth, trans_geo_smooth, rot_geo_smooth, overlap_geo_smooth = [], [], [], [], []

    for scene in tqdm(scenes, desc="Scenes"):
        gt_path = os.path.join(benchmark, scene)
        os.makedirs(gt_path, exist_ok=True)

        # Find all pairs for this scene
        scene_id = [k for k, s in enumerate(src) if os.path.basename(os.path.dirname(s)) == scene]
        src_i = src[scene_id]
        tgt_i = tgt[scene_id]
        rot_i = rot[scene_id, :, :]
        trans_i = trans[scene_id, :, :]
        overlap_i = overlap[scene_id]

        plane_id = []
        for pair in tqdm(range(len(src_i)), desc=f"Pairs in {scene}", leave=False):
            src_id = int(re.findall(r"\d+", os.path.basename(src_i[pair]))[0])
            tgt_id = int(re.findall(r"\d+", os.path.basename(tgt_i[pair]))[0])

            # Compose transformation matrix
            pose = np.zeros((4, 4))
            pose[:3, :3] = rot_i[pair]
            pose[:3, 3] = trans_i[pair].T
            pose[3, 3] = 1.0

            # Load and convert point clouds
            src_pcd = np.load(os.path.join(config.input_root, 'data', src_i[pair]))['xyz']
            tgt_pcd = np.load(os.path.join(config.input_root, 'data', tgt_i[pair]))['xyz']
            src_pcd = array2pcd(src_pcd)
            tgt_pcd = array2pcd(tgt_pcd)

            # Transform source point cloud
            src_pcd_temp = copy.deepcopy(src_pcd)
            tgt_pcd_temp = copy.deepcopy(tgt_pcd)
            src_pcd_temp = src_pcd_temp.transform(pose)

            # Find overlapping points
            src_pts, tgt_pts = src_pcd_temp.points, tgt_pcd_temp.points
            nn_distances, idx = get_nearest_neighbor(src_pts, tgt_pts, return_index=True)
            overlap_idx = nn_distances < 0.1
            pcd_idx = idx[overlap_idx]
            overlap_pts = pcd2array(tgt_pcd_temp, return_colors=False)[pcd_idx]

            # Fit a plane to the overlapping points
            plane = pyrsc.Plane()
            np.seterr(invalid='ignore')
            _, best_inliers = plane.fit(overlap_pts, 0.1)
            in_plane_ratio = len(best_inliers) / len(overlap_pts) if len(overlap_pts) > 0 else 0

            if in_plane_ratio < config.in_plane_threshold:
                continue

            print(f'{scene}, {pair}')
            print(src_i[pair], tgt_i[pair])

            plane_id.append(pair)
            with open(f"{gt_path}/gt.log", 'a') as f:
                f.write(f'{src_id}\t{tgt_id}\t{len(plane_id)}\n')
                for row in pose:
                    f.write('\t'.join(str(val) for val in row) + '\n')

        src_geo_smooth.append(src_i[plane_id])
        tgt_geo_smooth.append(tgt_i[plane_id])
        rot_geo_smooth.append(rot_i[plane_id, :, :])
        trans_geo_smooth.append(trans_i[plane_id, :, :])
        overlap_geo_smooth.append(overlap_i[plane_id])

    # Save filtered dataset
    data_2 = copy.deepcopy(data)
    data_2["src"] = np.concatenate(src_geo_smooth)
    data_2["tgt"] = np.concatenate(tgt_geo_smooth)
    data_2["rot"] = np.concatenate(rot_geo_smooth)
    data_2["trans"] = np.concatenate(trans_geo_smooth)
    data_2["overlap"] = np.concatenate(overlap_geo_smooth)

    out_meta_dir = osp.join(config.output_root, 'metadata')
    os.makedirs(out_meta_dir, exist_ok=True)
    with open(os.path.join(out_meta_dir, f'{dataset_name}_preprocess.pkl'), "wb") as f:
        pickle.dump(data_2, f)

    # Save GeoTransformer format .pkl file
    pkl_list = []
    for i in range(len(data_2['overlap'])):
        entry = {
            'overlap': data_2['overlap'][i],
            'pcd0': data_2['tgt'][i],
            'pcd1': data_2['src'][i],
            'rotation': data_2['rot'][i],
            'translation': data_2['trans'][i].squeeze(),
            'scene_name': os.path.basename(os.path.dirname(data_2['tgt'][i])),
            'frag_id0': int(re.findall(r'\d+', os.path.basename(data_2['tgt'][i]))[0]),
            'frag_id1': int(re.findall(r'\d+', os.path.basename(data_2['src'][i]))[0])
        }
        pkl_list.append(entry)

    save_path = osp.join(out_meta_dir, f"{dataset_name}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(pkl_list, f)

    print("Process is done!")

def main():
    parser = argparse.ArgumentParser("Create planar subset dataset based on ScanNet++")
    parser.add_argument('--config', type=str, default='./config/subset_scannetpp.yaml', help='Path to config file.')
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    config = edict(cfg)
    run_time(create_scannetpp_planar(config))

if __name__ == '__main__':
    main()
