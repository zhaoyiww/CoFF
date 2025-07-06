# Create a smaller planar subset of the IndoorLRS dataset for testing

import pickle
import copy
import os
import numpy as np
import re
from scipy.spatial import cKDTree
import open3d as o3d
from tqdm import tqdm

def pc2ndarray(point_cloud, color=True):
    """Convert Open3D point cloud to numpy array (optionally with color)."""
    src_pc = copy.deepcopy(point_cloud)
    src_points = np.asarray(src_pc.points)
    if color:
        src_colors = np.asarray(src_pc.colors)
        return src_points, src_colors
    return src_points

def ndarray2pc(point_array, colors=None, normals=None):
    """Convert numpy array to Open3D point cloud."""
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_array)
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        point_cloud.normals = o3d.utility.Vector3dVector(normals)
    return point_cloud

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
dataset = "IndoorRGBD_all_test"
mode = "all_test"
base_path = "./inputs/configs"
file_name = f"{mode}_info.pkl"

# Load data
with open(os.path.join(base_path, dataset, file_name), "rb") as f:
    data = pickle.load(f)

data["trans"] = np.array(data["trans"])
data["rot"] = np.array(data["rot"])
data["overlap"] = np.array(data["overlap"])

# Select all pairs (mask can be adjusted for overlap range)
mask = data['overlap'] < 1
lo_set = np.where(mask)

src = np.array(data['src'])[lo_set]
tgt = np.array(data['tgt'])[lo_set]
trans = np.array(data["trans"])[lo_set]
rot = np.array(data["rot"])[lo_set]
overlap = data['overlap'][lo_set]

# Scene configuration
scenes = ['apartment', 'bedroom', 'boardroom', 'lobby', 'loft']

benchmark = os.path.join(
    base_path.replace('inputs', 'outputs'),
    dataset,
    'benchmark/IndoorRGBD_test_all_scene_img_64_ol_01_10_geo_smooth_08'
)
os.makedirs(benchmark, exist_ok=True)

src_geo_smooth, tgt_geo_smooth, trans_geo_smooth, rot_geo_smooth, overlap_geo_smooth = [], [], [], [], []

for scene in tqdm(scenes, desc="Scenes"):
    gt_path = os.path.join(benchmark, scene)
    os.makedirs(gt_path, exist_ok=True)

    # Find all pairs belonging to this scene
    scene_id = [k for k, s in enumerate(src) if os.path.basename(os.path.dirname(s)) == scene]
    src_i = src[scene_id]
    tgt_i = tgt[scene_id]
    rot_i = rot[scene_id, :, :]
    trans_i = trans[scene_id, :, :]
    overlap_i = overlap[scene_id]

    plane_id = []
    for pair in tqdm(range(len(src_i)), desc=f"Pairs in {scene}"):
        src_id = int(re.findall(r"\d+", src_i[pair])[0])
        tgt_id = int(re.findall(r"\d+", tgt_i[pair])[0])

        # Compose transformation matrix
        pose = np.zeros((4, 4))
        pose[:3, :3] = rot_i[pair]
        pose[:3, 3] = trans_i[pair].T
        pose[3, 3] = 1.0

        # Load and convert point clouds
        src_pcd = np.load(os.path.join('inputs', src_i[pair]))['xyz']
        tgt_pcd = np.load(os.path.join('inputs', tgt_i[pair]))['xyz']
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
        print('in_plane_ratio:', in_plane_ratio)
        if in_plane_ratio < 0.8:
            continue

        plane_id.append(pair)
        with open(f"{gt_path}/gt.log", 'a') as f:
            f.write(f'{tgt_id}\t{src_id}\t{len(plane_id)}\n')
            for row in pose:
                f.write('\t'.join(str(val) for val in row) + '\n')

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

os.makedirs(os.path.join(base_path, dataset, "configs"), exist_ok=True)
with open(
    os.path.join(
        base_path.replace('inputs', 'outputs'),
        dataset,
        "configs",
        'IndoorLRS_geo_smooth_08.pkl'
    ),
    "wb"
) as f:
    pickle.dump(data_2, f)
    print("Process is done!")