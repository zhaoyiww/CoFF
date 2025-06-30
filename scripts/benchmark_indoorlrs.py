# Conpute metrics regarding the results of evaluate (modified from Predator)
# for IndoorLRS dataset
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from utils.metrics import compute_registration_rmse, compute_registration_error

# IndoorLRS, or IndoorLRS_planar
benchmark = 'IndoorLRS'
scenes = ["apartment", "bedroom", "boardroom", "lobby", "loft"]
mode = "test"


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

# # add flags
# all_flags, all_est_traj = [], []

for scene in scenes:
    gt_path = f"../data/IndoorLRS/metadata/benchmarks/{benchmark}/{scene}/gt.log"
    est_path = f"../outputs/IndoorLRS/registration/{benchmark}/{scene}/est.log"

    pairs, pose_gt = read_log(gt_path)
    sorted_indices = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs_sorted = pairs[sorted_indices]
    pose_gt_sorted = pose_gt[sorted_indices]

    _, pose_est = read_log(est_path)

    # result_dict = dict()
    registration_rmse_scene, rre_scene, rte_scene = [], [], []
    n_valid = pose_est.shape[0]

    # add flags
    flags_per_scene, est_traj_per_scene = [], []

    for i in tqdm(range(pose_est.shape[0])):
        # point cloud path
        tgt_pts_path = os.path.join(f"../data/IndoorLRS/data/test/{scene}", f"mesh_{pairs_sorted[i, 0]}.npz")
        tgt_pts = np.load(tgt_pts_path)["xyz"]

        registration_rmse = compute_registration_rmse(tgt_pts, pose_est[i], pose_gt_sorted[i])

        rre, rte = compute_registration_error(pose_est[i], pose_gt_sorted[i])

        registration_rmse_scene.append(registration_rmse)
        rre_scene.append(rre)
        rte_scene.append(rte)

        # add flags
        if registration_rmse <= 0.2:
            flags_per_scene.append(0)
        else:
            flags_per_scene.append(1)
        est_traj_per_scene.append(pose_est[i])

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

# # save all set
# reg_set = dict()
# reg_set['est_traj'] = all_est_traj
# reg_set['all_flags'] = all_flags
# import pickle
# with open(f'{benchmark}_reg_set.pkl', "wb") as b:
#     pickle.dump(reg_set, b)
#     print("all set file is saved!")
# b.close()

# print(f"registration_rmse_all:{np.array(registration_rmse_all)}")
# print(f"rre_all:{np.array(rre_all)}")
# print(f"rte_all:{np.array(rte_all)}")
# print(f"registration_recall_all:{np.array(registration_recall_all)}")

weighted_rmse = (np.array(n_valids) * np.array(registration_rmse_all['median'])).sum() / np.sum(n_valids)
weighted_rre = (np.array(n_valids) * np.array(rre_all['median'])).sum() / np.sum(n_valids)
weighted_rte = (np.array(n_valids) * np.array(rte_all['median'])).sum() / np.sum(n_valids)
weighted_rr = (np.array(n_valids) * np.array(registration_recall_all)).sum() / np.sum(n_valids)

print("====================================")
print(f"weighted mean registration_rmse_all:{np.mean(np.array(registration_rmse_all['mean'])):.3f}")
print(f"weighted mean rre_all:{np.mean(np.array(rre_all['mean'])):.3f}")
print(f"weighted mean rte_all:{np.mean(np.array(rte_all['mean'])):.3f}")
print(f"weighted registration_recall_all:{np.mean(np.array(registration_recall_all)):.3f}")

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