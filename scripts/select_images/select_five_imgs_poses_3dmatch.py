# 3DMatch
# select certain images and image poses, three images
import os
import math

import numpy as np
import pandas as pd
import shutil
from shutil import copy
from tqdm import tqdm
import re, glob


# source: predator/lib/natural_key
def natural_key(string_):
    """
    Sort strings by numbers in the name
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


mode = "val"
scenes = open(f'./inputs/3DMatch/{mode}_3dmatch.txt').readlines()
# scenes = ["7-scenes-redkitchen"]
# mode = "test"
base_path = "./inputs/3DMatch/Raw_Images"
save_path = "./outputs/3DMatch/five_images_per_fragment"

os.makedirs(save_path, exist_ok=True)
shutil.copy2('select_five_imgs_poses_3dmatch.py', save_path)

for scene in tqdm(scenes):
    scene = scene.strip('\n')
    save_img_dir = os.path.join(save_path, scene)
    os.makedirs(save_img_dir, exist_ok=True)

    if mode == 'val':
        pair_lists = sorted(glob.glob(f'./inputs/3DMatch/data/train/{scene}/*.txt'), key=natural_key)
    else:
        pair_lists = sorted(glob.glob(f'./inputs/3DMatch/data/{mode}/{scene}/*.txt'), key=natural_key)

    for pair in pair_lists:
        f = open(pair)
        info = f.readlines()
        id_seq = info[0].split()[1]
        id_start_fra = info[0].split()[2]
        id_end_fra = info[0].split()[3]
        id_mid_fra = str(int((int(id_end_fra) + int(id_start_fra)) / 2))

        # add two more images
        id_mid_num_2 = (int(id_end_fra) - int(id_start_fra)) // 4 + int(id_start_fra)
        id_last_num_2 = (int(id_end_fra) - int(id_start_fra)) // 4 * 3 + + int(id_start_fra)

        id_mid_num_2 = str(id_mid_num_2)
        id_last_num_2 = str(id_last_num_2)

        seq_dir = os.path.join(save_img_dir, id_seq)
        os.makedirs(seq_dir, exist_ok=True)

        filenames = [f"frame-{id_start_fra.zfill(6)}", f"frame-{id_mid_num_2.zfill(6)}",
                     f"frame-{id_mid_fra.zfill(6)}", f"frame-{id_last_num_2.zfill(6)}", f"frame-{id_end_fra.zfill(6)}"]
        # filenames = [f"frame-{id_start_fra.zfill(6)}", f"frame-{id_end_fra.zfill(6)}"]

        file_path = os.path.join(base_path, scene, id_seq)
        for filename in filenames:
            raw_path_clr = os.path.join(file_path, f"{filename}.color.png")
            if os.path.isfile(raw_path_clr) is False:
                raw_path_clr = os.path.join(file_path, f"{filename}.color.jpg")
                if os.path.isfile(raw_path_clr) is False:
                    file_path =file_path.replace(scene, str(scene.rsplit('_', 1)[0]))
                    raw_path_clr = os.path.join(file_path, f"{filename}.color.png")
                    scene = str(scene.rsplit('_', 1)[0])
                    if os.path.isfile(raw_path_clr) is False:
                        file_path = file_path.replace(scene, str(scene.rsplit('_', 1)[0]))
                        raw_path_clr = os.path.join(file_path, f"{filename}.color.png")
                        scene = str(scene.rsplit('_', 1)[0])

            raw_path_depth = os.path.join(file_path, f"{filename}.depth.png")
            raw_path_pose = os.path.join(file_path, f"{filename}.pose.txt")
            copy(raw_path_clr, seq_dir)
            copy(raw_path_depth, seq_dir)
            copy(raw_path_pose, seq_dir)
        f.close()
    instrinsic_path = os.path.join(base_path, scene, "camera-intrinsics.txt")
    copy(instrinsic_path, save_img_dir)

    print(f"scene: '{scene}' processing is done, and results are saved in '{save_path}'\n")
