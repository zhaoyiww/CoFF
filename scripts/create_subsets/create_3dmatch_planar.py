# Create a smaller planar subset of the 3DMatch dataset for testing

import pickle
import copy
import os
import numpy as np
import re
from scipy.spatial import cKDTree
import open3d as o3d
from tqdm import tqdm
import torch
from utils.o3d_tools import *

def get_nearest_neighbor(src_pts, tgt_pts, return_index=False):
    """Find nearest neighbors from tgt_pts to src_pts using KDTree."""
    s_tree = cKDTree(src_pts)
    dist, indices = s_tree.query(tgt_pts, k=1, workers=-1)
    if return_index:
        return dist, indices
    return dist

def cal_overlap(src_pcd, tgt_pcd, trans_matrix=None, overlap_radius=0.1):
    """Compute the overlap ratio between two point clouds."""
    src_pcd_temp = copy.deepcopy(src_pcd)
    tgt_pcd_temp = copy.deepcopy(tgt_pcd)
    if trans_matrix is not None:
        src_pcd = src_pcd_temp.transform(trans_matrix)
    src_pts, tgt_pts = src_pcd.points, tgt_pcd_temp.points
    nn_distances, _ = get_nearest_neighbor(src_pts, tgt_pts, return_index=True)
    overlap = np.mean(nn_distances < overlap_radius)
    return overlap

# Configuration
dataset = "3DMatch"
mode = "3DMatch"
base_path = "./inputs/configs"
file_name = f"{mode}.pkl"
in_plane_thre = 0.7

# Load data
with open(os.path.join(base_path, dataset, file_name), "rb") as f:
    data = pickle.load(f)

data["trans"] = np.array(data["trans"])
data["rot"] = np.array(data["rot"])
data["overlap"] = np.array(data["overlap"])

# Select all pairs (mask can be adjusted for overlap range)
mask = data['overlap'] <= 1
lo_set = np.where(mask)

src = np.array(data['src'])[lo_set]
tgt = np.array(data['tgt'])[lo_set]
trans = np.array(data["trans"])[lo_set]
rot = np.array(data["rot"])[lo_set]
overlap = data['overlap'][lo_set]

# Scene and fragment configuration
scenes = [
    '7-scenes-redkitchen', 'sun3d-home_at-home_at_scan1_2013_jan_1',
    'sun3d-home_md-home_md_scan9_2012_sep_30', 'sun3d-hotel_uc-scan3',
    'sun3d-hotel_umd-maryland_hotel1', 'sun3d-hotel_umd-maryland_hotel3',
    'sun3d-mit_76_studyroom-76-1studyroom2', 'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
]
num_frag = [60, 60, 60, 55, 57, 37, 66, 38]

dataset_name = f'{mode}_geo_smooth_{in_plane_thre}'
benchmark = os.path.join(base_path.replace('inputs', 'outputs'), dataset, f'benchmarks/{dataset_name}')
os.makedirs(benchmark, exist_ok=True)

# Prepare lists for filtered data
src_geo_smooth, tgt_geo_smooth, trans_geo_smooth, rot_geo_smooth, overlap_geo_smooth = [], [], [], [], []

for num, scene in tqdm(enumerate(scenes), desc="Scenes"):
    gt_path = os.path.join(benchmark, scene)
    os.makedirs(gt_path, exist_ok=True)

    # Find all pairs belonging to this scene
    scene_id = [k for k, s in enumerate(src) if os.path.basename(os.path.dirname(s)) == scene]
    src_i = src[scene_id]
    tgt_i = tgt[scene_id]
    rot_i = rot[scene_id, :, :]
    trans_i = trans[scene_id, :, :]
    overlap_i = overlap[scene_id]

    # Load ground truth info for this scene
    gt_info_path = os.path.join(base_path, dataset, 'raw_gt_info', mode, scene, 'gt.info')
    with open(gt_info_path, 'r') as g:
        gt_info = g.readlines()
    gt_info_0 = copy.deepcopy(gt_info)

    plane_id = []
    for pair in tqdm(range(len(src_i)), desc=f"Pairs in {scene}"):
        src_id = int(re.findall(r"\d+", os.path.basename(src_i[pair]))[0])
        tgt_id = int(re.findall(r"\d+", os.path.basename(tgt_i[pair]))[0])

        # Compose transformation matrix
        pose = np.zeros((4, 4))
        pose[:3, :3] = rot_i[pair]
        pose[:3, 3] = trans_i[pair].T
        pose[3, 3] = 1.0

        # Load and convert point clouds
        src_pcd = torch.load(os.path.join('inputs', src_i[pair]))
        tgt_pcd = torch.load(os.path.join('inputs', tgt_i[pair]))
        src_pcd = ndarray2pc(src_pcd)
        tgt_pcd = ndarray2pc(tgt_pcd)

        # Transform source point cloud
        src_pcd_temp = copy.deepcopy(src_pcd)
        tgt_pcd_temp = copy.deepcopy(tgt_pcd)
        src_pcd_temp = src_pcd_temp.transform(pose)

        # Find overlapping points
        src_pts, tgt_pts = src_pcd_temp.points, tgt_pcd_temp.points
        nn_distances, idx = get_nearest_neighbor(tgt_pts, src_pts, return_index=True)
        overlap_idx = nn_distances < 0.1
        pcd_idx = idx[overlap_idx]
        overlap_pts = pc2ndarray(tgt_pcd_temp, color=False)[pcd_idx]

        # Fit a plane to the overlapping points
        import pyransac3d as pyrsc
        plane = pyrsc.Plane()
        np.seterr(invalid='ignore')
        _, best_inliers = plane.fit(overlap_pts, 0.1)

        in_plane_ratio = len(best_inliers) / len(overlap_pts)
        if in_plane_ratio < in_plane_thre:
            continue

        print(f'{scene}, {pair}')
        print(src_i[pair], tgt_i[pair])

        # Find the corresponding gt.info entry
        gt_info_pair_info = gt_info[0:-1:7]
        gt_info_pair_info = [item.replace('\t', ' ') for item in gt_info_pair_info]
        gt_info_pair_info = [item.replace('\n', ' ') for item in gt_info_pair_info]
        gt_info_pair_info = [item.replace('  ', ' ') for item in gt_info_pair_info]
        gt_idx = np.where(np.array(gt_info_pair_info) == f'{tgt_id} {src_id} {num_frag[num]} ')[0]

        # Save gt.info and gt.log for this pair
        with open(f"{gt_path}/gt.info", 'a') as ff:
            ff.writelines(gt_info[int(gt_idx[0]) * 7: int(gt_idx[0]) * 7 + 7])

        plane_id.append(pair)
        with open(f"{gt_path}/gt.log", 'a') as f:
            f.write(f'{tgt_id}\t{src_id}\t{len(plane_id)}\n')
            for row in pose:
                f.write('\t'.join(str(val) for val in row) + '\n')

    # Collect filtered pairs for this scene
    src_geo_smooth.append(src_i[plane_id])
    tgt_geo_smooth.append(tgt_i[plane_id])
    rot_geo_smooth.append(rot_i[plane_id, :, :])
    trans_geo_smooth.append(trans_i[plane_id, :, :])
    overlap_geo_smooth.append(overlap_i[plane_id])

# Save the filtered dataset
data_2 = copy.deepcopy(data)
data_2["src"] = np.concatenate(src_geo_smooth)
data_2["tgt"] = np.concatenate(tgt_geo_smooth)
data_2["rot"] = np.concatenate(rot_geo_smooth)
data_2["trans"] = np.concatenate(trans_geo_smooth)
data_2["overlap"] = np.concatenate(overlap_geo_smooth)

os.makedirs(os.path.join(base_path, dataset, "3DMatch"), exist_ok=True)
with open(os.path.join(base_path.replace('inputs', 'outputs'), dataset, "3DMatch", f'{dataset_name}.pkl'), "wb") as f:
    pickle.dump(data_2, f)
    print("Process is done!")