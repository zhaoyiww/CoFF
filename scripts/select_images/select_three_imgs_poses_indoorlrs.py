# ---------------------------------------------------------------------------- #
# IndoorLRS Image Preparation Script
# select three images and image poses for each point cloud fragment
# ---------------------------------------------------------------------------- #
import os
import os.path as osp
import math
import pandas as pd
import shutil
from shutil import copy
from tqdm import tqdm


# Configuration
scenes = ["apartment", "bedroom", "boardroom", "lobby", "loft"]
img_type = 'depth'
base_path = "./IndoorLRS_ImgPreparation/Input_Images"
save_path = "./IndoorLRS_ImgPreparation/Outputs/images-three-per-fragments"

# Create output directory
os.makedirs(save_path, exist_ok=True)
shutil.copy2('select_three_imgs_poses_indoorlrs.py', save_path)

# Process each scene
for scene in scenes:
    save_img_dir = osp.join(save_path, scene, img_type)
    os.makedirs(save_img_dir, exist_ok=True)

    file_path = osp.join(base_path, scene, img_type)
    pose_path = osp.join(base_path, scene, f"{scene}.log")

    # Check if required directories and files exist
    if not osp.exists(file_path):
        print(f"Warning: Image directory not found for scene '{scene}'")
        continue
    if not osp.isfile(pose_path):
        print(f"Warning: Pose file not found for scene '{scene}'")
        continue

    # Get sorted list of image files
    pathDir = [x for x in os.listdir(file_path) if x.endswith(('.jpg', '.png'))]
    pathDir = sorted(pathDir)

    pose_all = pd.read_csv(pose_path, header=None)
    frag_pcd_pose_path = pose_path.replace('apartment.log', 'frag_pose/pose_slac.log')
    if not osp.isfile(frag_pcd_pose_path):
        print(f"Warning: Fragment pose file not found for scene '{scene}'")
        continue
    frag_pcd_pose = pd.read_csv(frag_pcd_pose_path, header=None)

    interval = 100
    for k in tqdm(range(math.ceil(len(pathDir) / interval))):
        # Determine indices for start, middle, and end frames
        if k == math.ceil((len(pathDir) - 1) / interval) - 1:
            last_num = (len(pathDir) - 1) % interval
            mid_num = math.floor(last_num / 2)
        else:
            last_num = 99
            mid_num = 49

        filenames = [pathDir[k * interval], pathDir[k * interval + mid_num], pathDir[k * interval + last_num]]

        # Copy image files
        for filename in filenames:
            raw_path = osp.join(file_path, filename)
            if osp.isfile(raw_path):
                copy(raw_path, save_img_dir)
            else:
                print(f"Warning: File '{filename}' not found in '{file_path}'")

        # Extract and save pose data
        pose_0 = pose_all[k * interval * 5: k * interval * 5 + 5]
        pose_1 = pose_all[(k * interval + mid_num) * 5: (k * interval + mid_num) * 5 + 5]
        pose_2 = pose_all[(k * interval + last_num) * 5: (k * interval + last_num) * 5 + 5]

        pose_dir = save_img_dir.replace(f'/{img_type}', '/pose')
        os.makedirs(pose_dir, exist_ok=True)
        pd.DataFrame(pose_0[1:]).to_csv(osp.join(pose_dir, f"{osp.splitext(filenames[0])[0]}.pose.txt"), header=False, index=False)
        pd.DataFrame(pose_1[1:]).to_csv(osp.join(pose_dir, f"{osp.splitext(filenames[1])[0]}.pose.txt"), header=False, index=False)
        pd.DataFrame(pose_2[1:]).to_csv(osp.join(pose_dir, f"{osp.splitext(filenames[2])[0]}.pose.txt"), header=False, index=False)

        # Save fragment pose info
        info_title = f"{scene}\t{int(osp.splitext(filenames[0])[0])}\t{int(osp.splitext(filenames[2])[0])}"
        frag_pose_curr = frag_pcd_pose[k * 5: k * 5 + 5].values[1:]
        frag_pose_dir = save_img_dir.replace(f'/{img_type}', '/frag_pcd_pose')
        os.makedirs(frag_pose_dir, exist_ok=True)

        with open(osp.join(frag_pose_dir, f"mesh_{int(osp.splitext(filenames[0])[0]) / 100:.0f}.info.txt"), 'w') as log_file:
            log_file.write(info_title + "\n")
            for line in frag_pose_curr:
                clean_line = str(line).replace('[', '').replace(']', '').replace('\'', '')
                log_file.write(clean_line + "\n")

    print(f"Scene: '{scene}' processing is done, and results are saved in '{save_path}'")