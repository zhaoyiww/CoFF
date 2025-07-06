# ---------------------------------------------------------------------------- #
# ScanNetpp Image Preparation Script
# select three images and image poses for each point cloud fragment
# ---------------------------------------------------------------------------- #
import os
import shutil
from shutil import copy
from tqdm import tqdm
import re
import glob
import os.path as osp


# Function to sort strings by numbers in the name
def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


# Configuration
dataset = 'ScanNetpp'
mode = "test"
scenes_file = f'../../data/{dataset}/metadata/split/{mode}_scannetpp.txt'
base_path = f"./inputs/{dataset}/Raw_Images"
save_path = f"./outputs/{dataset}/three_images_per_fragment"

# Ensure output directory exists
os.makedirs(save_path, exist_ok=True)
shutil.copy2('copy_img_pose_three-imgs_scannetpp.py', save_path)

# Read scene list
if not osp.isfile(scenes_file):
    raise FileNotFoundError(f"Scenes file '{scenes_file}' not found.")
scenes = open(scenes_file).readlines()

# Process each scene
for scene in tqdm(scenes):
    scene = scene.strip()
    save_img_dir = osp.join(save_path, scene)
    os.makedirs(save_img_dir, exist_ok=True)

    # Get list of pair files
    pair_lists = sorted(glob.glob(f'./inputs/{dataset}/data/{mode}/{scene}/*.txt'), key=natural_key)
    if not pair_lists:
        print(f"Warning: No pair files found for scene '{scene}'.")
        continue

    for pair in pair_lists:
        with open(pair) as f:
            info = f.readlines()
            id_seq = info[0].split()[1]
            id_start_fra = info[0].split()[2]
            id_end_fra = info[0].split()[3]
            id_mid_fra = str(int((int(id_end_fra) + int(id_start_fra)) / 2))

        # Create directory for RGB images
        seq_dir = osp.join(save_img_dir, 'rgb')
        os.makedirs(seq_dir, exist_ok=True)

        # Define filenames for start, middle, and end frames
        filenames = [f"frame_{id_start_fra.zfill(6)}", f"frame_{id_mid_fra.zfill(6)}", f"frame_{id_end_fra.zfill(6)}"]

        # Copy RGB images
        file_path = osp.join(base_path, scene)
        for filename in filenames:
            raw_path_clr = osp.join(file_path, 'rgb', f"{filename}.jpg")
            if not osp.isfile(raw_path_clr):
                print(f"Warning: RGB file '{raw_path_clr}' not found.")
                continue
            copy(raw_path_clr, seq_dir)

    # Copy intrinsic file
    intrinsic_path = osp.join(base_path, scene, "pose_intrinsic_imu.json")
    if osp.isfile(intrinsic_path):
        copy(intrinsic_path, save_img_dir)
    else:
        print(f"Warning: Intrinsic file '{intrinsic_path}' not found for scene '{scene}'.")

print(f"Number of scenes: {len(scenes)}. Processing is done, and results are saved in '{save_path}'.")