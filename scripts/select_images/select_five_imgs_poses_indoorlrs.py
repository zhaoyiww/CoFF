# select certain images and image poses, three images
import os
import math

import numpy as np
import pandas as pd
import shutil
from shutil import copy
from tqdm import tqdm


scenes = ["apartment", "bedroom", "boardroom", "lobby", "loft"]
# scenes = ["apartment"]
# rgb or depth
img_type = 'rgb'
# img_type_2 = 'depth'
base_path = "./IndoorRGBD_ImgPreparation/Input_Images"
save_path = "./IndoorRGBD_ImgPreparation/Outputs/images-five-per-fragments"

os.makedirs(save_path, exist_ok=True)
shutil.copy2('select_five_imgs_poses_indoorlrs.py', save_path)

for scene in scenes:
    save_img_dir = os.path.join(save_path, scene, img_type)
    os.makedirs(save_img_dir, exist_ok=True)

    file_path = os.path.join(base_path, scene, img_type)
    pose_path = os.path.join(base_path, scene, f"{scene}.log")

    pathDir = os.listdir(file_path)
    if os.path.splitext(pathDir[0])[1] == '.jpg':
        pathDir = [x for x in pathDir if x.endswith('.jpg')]
    else:
        pathDir = [x for x in pathDir if x.endswith('.png')]
    pathDir = sorted(pathDir)

    pose_all = pd.read_csv(pose_path, header=None)
    pose_export = pd.DataFrame([])

    frag_pcd_pose = pd.read_csv(pose_path.replace('apartment.log', 'frag_pose/pose_slac.log'), header=None)

    interval = 100
    for k in tqdm(range(math.ceil(len(pathDir) / interval))):
        if k == math.ceil((len(pathDir) - 1) / interval) - 1:
            last_num = (len(pathDir) - 1) % interval
            mid_num = math.floor(last_num / 2)
        else:
            last_num = 99
            mid_num = 49

        mid_num_2 = last_num // 4
        last_num_2 = mid_num_2 * 3

        filenames = [pathDir[k * interval], pathDir[k * interval + mid_num], pathDir[k * interval + last_num],
                     pathDir[k * interval + mid_num_2], pathDir[k * interval + last_num_2]]

        for filename in filenames:
            raw_path = os.path.join(file_path, filename)
            # copy img files
            copy(raw_path, save_img_dir)

        # index = np.where(id_location == int(indices[i]))[0][0]
        pose_0 = pose_all[k * interval * 5: k * interval * 5 + 5][:]
        pose_1 = pose_all[(k * interval + mid_num) * 5: (k * interval + mid_num) * 5 + 5][:]
        pose_2 = pose_all[(k * interval + last_num) * 5: (k * interval + last_num) * 5 + 5][:]

        pose_3 = pose_all[(k * interval + mid_num) * 5: (k * interval + mid_num_2) * 5 + 5][:]
        pose_4 = pose_all[(k * interval + last_num) * 5: (k * interval + last_num_2) * 5 + 5][:]

        os.makedirs(save_img_dir.replace(f'/{img_type}', '/pose'), exist_ok=True)
        pd.DataFrame(pose_0[1:]).to_csv(os.path.join(save_img_dir.replace(f'/{img_type}', '/pose'), f"{os.path.splitext(filenames[0])[0]}.pose.txt"), header=False, index=False)
        pd.DataFrame(pose_1[1:]).to_csv(os.path.join(save_img_dir.replace(f'/{img_type}', '/pose'), f"{os.path.splitext(filenames[1])[0]}.pose.txt"), header=False, index=False)
        pd.DataFrame(pose_2[1:]).to_csv(os.path.join(save_img_dir.replace(f'/{img_type}', '/pose'), f"{os.path.splitext(filenames[2])[0]}.pose.txt"), header=False, index=False)

        pd.DataFrame(pose_3[1:]).to_csv(os.path.join(save_img_dir.replace(f'/{img_type}', '/pose'), f"{os.path.splitext(filenames[3])[0]}.pose.txt"), header=False, index=False)
        pd.DataFrame(pose_4[1:]).to_csv(os.path.join(save_img_dir.replace(f'/{img_type}', '/pose'), f"{os.path.splitext(filenames[4])[0]}.pose.txt"), header=False, index=False)

        # info.txt
        info_title = [scene + '	 ' + f'{int(os.path.splitext(filenames[0])[0])}' + '	 ' + f'{int(os.path.splitext(filenames[2])[0])}'][0]
        frag_pose_curr = frag_pcd_pose[k * 5: k * 5 + 5].values[1:]
        save_frag_pcd_pose_dir = save_img_dir.replace(f'/{img_type}', '/frag_pcd_pose')
        os.makedirs(save_frag_pcd_pose_dir, exist_ok=True)

        with open(os.path.join(save_frag_pcd_pose_dir, f'mesh_{int(os.path.splitext(filenames[0])[0]) / 100:.0f}.info.txt'), 'w') as log_file:
            log_file.write(info_title + "\n")
            # log_file.write(str(frag_pose_curr) + "\n")
            for line in str(frag_pose_curr):
                clean_line = line.replace('[', '').replace(']', '').replace('\'', '')
                log_file.write(clean_line)
                # log_file.write(line)

    #     if k == 0:
    #         pose_export = np.concatenate((pose_0, pose_1, pose_2), axis=0)
    #     else:
    #         pose_export = np.concatenate((pose_export, pose_0, pose_1, pose_2), axis=0)
    #
    # pd.DataFrame(pose_export).to_csv(os.path.join(save_path, f"{scene}_pose.log"), header=False, index=False)
    print(f"scene: '{scene}' processing is done, and results are saved in '{save_path}'")