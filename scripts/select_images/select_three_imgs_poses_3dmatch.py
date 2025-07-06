# ---------------------------------------------------------------------------- #
# 3DMatch Image Preparation Script
# select three images and image poses for each point cloud fragment
# ---------------------------------------------------------------------------- #
import os
import shutil
from shutil import copy
from tqdm import tqdm
import re
import glob
import os.path as osp


# Source: predator/lib/natural_key
def natural_key(string_):
    """
    Sort strings by numbers in the name
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


# Configuration
# "train", "val", or "test"
mode = "train"
scenes = open(f'../../data/3DMatch/metadata/split/{mode}_3dmatch.txt').readlines()

base_path = "./inputs/3DMatch/Raw_Images"
save_path = "./outputs/3DMatch/three_images_per_fragment"

# Create output directory
os.makedirs(save_path, exist_ok=True)
shutil.copy2('select_three_imgs_poses_3dmatch.py', save_path)

# Process each scene
for scene in tqdm(scenes):
    scene = scene.strip('\n')
    save_img_dir = osp.join(save_path, scene)
    os.makedirs(save_img_dir, exist_ok=True)

    pair_lists = sorted(glob.glob(f'./inputs/3DMatch/data/{mode}/{scene}/*.txt'), key=natural_key)

    for pair in pair_lists:
        with open(pair) as f:
            info = f.readlines()
            id_seq = info[0].split()[1]
            id_start_fra = info[0].split()[2]
            id_end_fra = info[0].split()[3]
            id_mid_fra = str(int((int(id_end_fra) + int(id_start_fra)) / 2))

        seq_dir = osp.join(save_img_dir, id_seq)
        os.makedirs(seq_dir, exist_ok=True)

        filenames = [f"frame-{id_start_fra.zfill(6)}", f"frame-{id_mid_fra.zfill(6)}", f"frame-{id_end_fra.zfill(6)}"]

        file_path = osp.join(base_path, scene, id_seq)
        for filename in filenames:
            # Handle color image
            raw_path_clr = osp.join(file_path, f"{filename}.color.png")
            if not osp.isfile(raw_path_clr):
                raw_path_clr = osp.join(file_path, f"{filename}.color.jpg")
            if not osp.isfile(raw_path_clr):
                print(f"Warning: Color image not found for {filename}")
                continue

            # Handle depth image
            raw_path_depth = osp.join(file_path, f"{filename}.depth.png")
            if not osp.isfile(raw_path_depth):
                print(f"Warning: Depth image not found for {filename}")
                continue

            # Handle pose file
            raw_path_pose = osp.join(file_path, f"{filename}.pose.txt")
            if not osp.isfile(raw_path_pose):
                print(f"Warning: Pose file not found for {filename}")
                continue

            # Copy files
            copy(raw_path_clr, seq_dir)
            copy(raw_path_depth, seq_dir)
            copy(raw_path_pose, seq_dir)

    # Copy intrinsic file
    intrinsic_path = osp.join(base_path, scene, "camera-intrinsics.txt")
    if osp.isfile(intrinsic_path):
        copy(intrinsic_path, save_img_dir)
    else:
        print(f"Warning: Intrinsic file not found for scene {scene}")

    print(f"Scene: '{scene}' processing is done, and results are saved in '{save_path}'\n")