# ---------------------------------------------------------------------------- #
# Compute metrics regarding the results of evaluation for the ScanNetpp dataset
# ---------------------------------------------------------------------------- #
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from utils.metrics import compute_registration_rmse, compute_registration_error
import os.path as osp

test_path_name = 'ScanNetpp'

# ScanNetpp_test, ScanNetpp_test_planar
benchmark = 'ScanNetpp_test_planar'
scene_path = osp.join('../data/ScanNetpp/metadata/split', 'test_scannetpp.txt')
f = open(scene_path)
scene_list = f.readlines()
scenes = [line.rstrip('\n') for line in scene_list]
f.close()


def read_log(filepath):
    pairs = pd.read_csv(filepath, skiprows=lambda x: x % 5 != 0, sep="\s+", na_values="-1", header=None)
    pairs = np.asarray(pairs)
    pairs = pairs[:, :2]

    pose = pd.read_csv(filepath, skiprows=lambda x: x % 5 == 0, sep="\s+", header=None)
    pose = np.asarray(pose)
    pose = pose.reshape(-1, 4, 4)
    return pairs, pose

from collections import defaultdict

registration_rmse_all, rre_all, rte_all = defaultdict(list), defaultdict(list), defaultdict(list)
registration_recall_all = []
n_valids = []

# add flags
all_flags, all_est_traj = [], []

re_single_all, te_single_all = [], []

for scene in scenes:
    gt_path = f"../data/ScanNetpp/metadata/benchmarks/{benchmark}/{scene}/gt.log"
    est_path = f"../outputs/{test_path_name}/registration/{benchmark}/{scene}/est.log"

    # sort gt.log
    pairs, pose_gt = read_log(gt_path)
    sorted_indices = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs_sorted = pairs[sorted_indices]
    pose_gt_sorted = pose_gt[sorted_indices]

    pairs_2, pose_est = read_log(est_path)
    sorted_indices_2 = np.lexsort((pairs_2[:, 1], pairs_2[:, 0]))
    pairs_sorted_2 = pairs_2[sorted_indices_2]
    pose_est_sorted = pose_est[sorted_indices_2]

    # result_dict = dict()
    registration_rmse_scene, rre_scene, rte_scene = [], [], []
    n_valid = pose_est_sorted.shape[0]

    # add flags
    flags_per_scene, est_traj_per_scene = [], []

    for i in tqdm(range(pose_est_sorted.shape[0])):
        # point cloud path
        src_pts_path = os.path.join(f"../data/ScanNetpp/data/test/{scene}", f"cloud_bin_{pairs_sorted[i, 0]}.npz")
        src_pts = np.load(src_pts_path)["xyz"]

        registration_rmse = compute_registration_rmse(src_pts, pose_est_sorted[i], pose_gt_sorted[i])

        rre, rte = compute_registration_error(pose_est_sorted[i], pose_gt_sorted[i])

        registration_rmse_scene.append(registration_rmse)
        rre_scene.append(rre)
        rte_scene.append(rte)

        # add flags
        if registration_rmse <= 0.2:
            flags_per_scene.append(0)
        else:
            flags_per_scene.append(1)
        est_traj_per_scene.append(pose_est_sorted[i])

    n_valids.append(n_valid)

    # registration_rmse_all.append(np.mean(registration_rmse_scene))
    # rre_all.append(np.mean(rre_scene))
    # rte_all.append(np.mean(rte_scene))
    registration_recall_all.append((np.sum(np.array(registration_rmse_scene) <= 0.2)) / n_valid)

    registration_rmse_all['mean'].append(np.mean(registration_rmse_scene))
    registration_rmse_all['median'].append(np.median(registration_rmse_scene))

    rte_all['mean'].append(np.mean(rte_scene))
    rte_all['median'].append(np.median(rte_scene))

    rre_all['mean'].append(np.mean(rre_scene))
    rre_all['median'].append(np.median(rre_scene))

    # # add flags
    # all_flags.append(flags_per_scene)
    # all_est_traj.append(np.array(est_traj_per_scene))
    #
    # # save all re and te
    # re_single_all.append(np.asarray(rre_scene))
    # te_single_all.append(np.asarray(rte_scene))

# # save all set, for qualitative comparison
# reg_set = dict()
# reg_set['est_traj'] = all_est_traj
# reg_set['all_flags'] = all_flags
# import pickle
# with open(f'est_transform_and_nonzero_flags_{benchmark}.pkl', "wb") as b:
#     pickle.dump(reg_set, b)
#     print("all set file is saved!")
# b.close()

# with open(f"re_{benchmark}.txt", "w") as file:
#     re_single_all = np.concatenate(re_single_all)
#     for item in re_single_all:
#         file.write(str(item) + "\n")
#
# with open(f"te_{benchmark}.txt", "w") as file:
#     te_single_all = np.concatenate(te_single_all)
#     for item in te_single_all:
#         file.write(str(item) + "\n")

# print(f"registration_rmse_all:{np.array(registration_rmse_all)}")
# print(f"rre_all:{np.array(rre_all)}")
# print(f"rte_all:{np.array(rte_all)}")
# print(f"registration_recall_all:{np.array(registration_recall_all)}")

weighted_rmse = (np.array(n_valids) * np.array(registration_rmse_all['median'])).sum() / np.sum(n_valids)
weighted_re = (np.array(n_valids) * np.array(rre_all['median'])).sum() / np.sum(n_valids)
weighted_te = (np.array(n_valids) * np.array(rte_all['median'])).sum() / np.sum(n_valids)
weighted_rr = (np.array(n_valids) * np.array(registration_recall_all)).sum() / np.sum(n_valids)

# average all scenes, balance the variety of scenes. We report this metric in the paper
print("====================================")
print(f"weighted mean registration_rmse_all:{np.mean(np.array(registration_rmse_all['mean'])):.3f}")
print(f"weighted mean rre_all:{np.mean(np.array(rre_all['mean'])):.3f}")
print(f"weighted mean rte_all:{np.mean(np.array(rte_all['mean'])):.3f}")
print(f"weighted registration_recall_all:{np.mean(np.array(registration_recall_all)):.3f}")

# # new one, average all pairs
# print("====================================")
# print(f"Evaluation Results on {benchmark}")
# print(f"total pairs for evaluation: {np.sum(n_valids)}")
# print(f"weighted mean re_all:{weighted_re:.3f} degree")
# print(f"weighted mean te_all:{weighted_te:.3f} m")
# print(f"weighted mean registration_rmse_all:{weighted_rmse:.3f} m")
# print(f"weighted registration_recall_all:{weighted_rr:.3f}")
# print("====================================")

# weighted_rre = (np.array(n_valids) * np.array(rre_all['mean'])).sum() / np.sum(n_valids)
# weighted_rte = (np.array(n_valids) * np.array(rte_all['mean'])).sum() / np.sum(n_valids)
# weighted_rmse = (np.array(n_valids) * np.array(registration_rmse_all['mean'])).sum() / np.sum(n_valids)
# weighted_rr = (np.array(n_valids) * np.array(registration_recall_all)).sum() / np.sum(n_valids)
#
# print("====================================")
# print(f"Evaluation Results on {benchmark}")
# print(f"total pairs for evaluation: {np.sum(n_valids)}")
# print(f"weighted mean rre_all:{weighted_rre:.3f} degree")
# print(f"weighted mean rte_all:{weighted_rte:.3f} m")
# print(f"weighted mean registration_rmse_all:{weighted_rmse:.3f} m")
# print(f"weighted registration_recall_all:{weighted_rr:.3f}")
# print("====================================")

# with open(f"snapshot/{test_path_name}/test/est_traj_/{dataset}/{num_keypts}/result-2", 'w') as f:
#     f.write(f"num. of pairs per scene: {n_valids}\n")
#     f.write(f"mean registration_rmse_all:{np.array(registration_rmse_all['mean'])}\n")
#     f.write(f"mean rre_all:{np.array(rre_all['mean'])}\n")
#     f.write(f"mean rte_all:{np.array(rte_all['mean'])}\n")
#     f.write(f"registration_recall_all:{np.array(registration_recall_all)}\n")
#     f.write("====================================\n")
#     f.write(f"median registration_rmse_all:{np.array(registration_rmse_all['median'])}\n")
#     f.write(f"median rre_all:{np.array(rre_all['median'])}\n")
#     f.write(f"median rte_all:{np.array(rte_all['median'])}\n")
#     f.write(f"registration_recall_all:{np.array(registration_recall_all)}\n")
#
#     f.write("====================================\n")
#     f.write(f"weighted mean registration_rmse_all:{np.mean(np.array(registration_rmse_all['mean'])):.3f}\n")
#     f.write(f"weighted mean rre_all:{np.mean(np.array(rre_all['mean'])):.3f}\n")
#     f.write(f"weighted mean rte_all:{np.mean(np.array(rte_all['mean'])):.3f}\n")
#     f.write(f"weighted registration_recall_all:{np.mean(np.array(registration_recall_all)):.3f}\n")

    # f.write("====================================\n")
    # f.write(f"weighted median registration_rmse_all:{np.mean(np.array(registration_rmse_all['median'])):.3f}\n")
    # f.write(f"weighted median rre_all:{np.mean(np.array(rre_all['median'])):.3f}\n")
    # f.write(f"weighted median rte_all:{np.mean(np.array(rte_all['median'])):.3f}\n")
    # f.write(f"weighted registration_recall_all:{np.mean(np.array(registration_recall_all)):.3f}\n")

# np.savetxt(os.path.join(folder, f"Registration_rmse_{args.scene}.txt"), registration_rmse_all, fmt="%.3f")