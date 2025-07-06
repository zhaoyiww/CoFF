import copy
import numpy as np
from scipy.spatial import cKDTree

def get_nearest_neighbor(src_pts, tgt_pts, return_index=False):
    s_tree = cKDTree(src_pts)
    dist, indices = s_tree.query(tgt_pts, k=1, workers=-1)
    if return_index:
        return dist, indices
    else:
        return dist

# source: geotransformer
def cal_overlap(src_pcd, tgt_pcd, trans_matrix=None, overlap_radius=0.1):
    r"""Compute the overlap of two point clouds."""
    src_pcd_temp = copy.deepcopy(src_pcd)
    tgt_pcd_temp = copy.deepcopy(tgt_pcd)
    if trans_matrix is not None:
        src_pcd = src_pcd_temp.transform(trans_matrix)
    src_pts, tgt_pts = src_pcd.points, tgt_pcd_temp.points
    nn_distances, idx = get_nearest_neighbor(src_pts, tgt_pts, return_index=True)
    overlap = np.mean(nn_distances < overlap_radius)
    return overlap