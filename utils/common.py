import os
import time
import open3d as o3d
import torch
import yaml
import numpy as np
import os.path as osp
import random
from scipy.spatial.transform import Rotation


def dir_exist(path, sub_folders=None):
    os.makedirs(path, exist_ok=True)
    if sub_folders is not None:
        for sub_folder in sub_folders:
            os.makedirs(osp.join(path, sub_folder), exist_ok=True)


def load_yaml(path, keep_sub_directory=False):
    """
    Load a '.yaml' config file,
    source from: [CoFiNet](https://github.com/haoyu94/Coarse-to-fine-correspondences/blob/main/configs/config_utils.py)
    :param keep_sub_directory:
    :param path:  path to config file
    :return: a dict of configuration parameters, merge sub_dicts
    """
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    if keep_sub_directory:
        return cfg
    else:
        config = dict()
        for key, value in cfg.items():
            if value:
                for k, v in value.items():
                    config[k] = v
        return config


def read_coord_without_name(file_path, delimiter=', '):
    """
    Read 3D coordinates from a file and return without names
    :param file_path:
    :param delimiter: define the delimiter between column
    :return:
    """

    # Create an empty dictionary to store the data for each point
    point_data = []

    # Open the file and read its contents
    with open(file_path, 'r') as file:
        # Read each line in the file
        for i, line in enumerate(file):
            # Split the line by commas to separate the values
            values = line.strip().split(delimiter)

            # The first value is the point name
            point_name = values[0]

            # The rest of the values are the data for that point
            data = list(map(float, values[1:]))

            # Store the data in the dictionary
            point_data.append(np.array(data))
    return np.array(point_data)


def rot_tran2transform(R, t):
    """
    Convert separate rotation matrix and translation vector to transformation matrix
    :param R: rotation matrix, dim: [3, 3]
    :param t: translation vector, dim: [3, 1]
    :return: transformation matrix, dim: [4, 4]
    """

    # Create a 4x4 identity matrix
    transform = np.eye(4)

    # Set the rotation part (top-left 3x3)
    transform[:3, :3] = R

    # Set the translation part (top-right 3x1)
    transform[:3, 3] = t.flatten()

    return transform


def transfrom2_rot_tran(transform):
    """
    Get rotation matrix and translation vector from rigid transform matrix.

    Args:
        transform (array): (4, 4)

    Returns:
        rotation (array): (3, 3)
        translation (array): (3,)
    """
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return rotation, translation


# def array2tensor(point_array):
#     """ Convert numpy array to torch tensor """
#     if not isinstance(point_array, torch.Tensor):
#         return torch.from_numpy(point_array).float()


def access_device():
    """ Access GPU if available, otherwise use CPU """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def min_max_normalize(array):
    """
    Normalize a numpy array to [0, 1]
    :param array:
    :return:
    """
    array_normalized = (array - array.min()) / (array.max() - array.min())
    return array_normalized


def read_pcd(pcd_path):
    return o3d.io.read_point_cloud(pcd_path)


def run_time(func1):
    start_time = time.time()
    func1
    end_time = time.time()
    return print(f'Time spent on {func1}: {end_time - start_time} seconds.')


# PointDsc/evaluation/benchmark_utils.py
# set the random seed for reproducing the results
def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_dict2path(file_path, result_dict):
    with open(file_path, 'w') as f:
        for key, value in result_dict.items():
            f.write(f"{key}: {value}\n")

# Source: GeoTrans
def get_log_string(result_dict, epoch=None, max_epoch=None, iteration=None, max_iteration=None, lr=None, timer=None):
    log_strings = []
    if epoch is not None:
        epoch_string = f'Epoch: {epoch}'
        if max_epoch is not None:
            epoch_string += f'/{max_epoch}'
        log_strings.append(epoch_string)
    if iteration is not None:
        iter_string = f'iter: {iteration}'
        if max_iteration is not None:
            iter_string += f'/{max_iteration}'
        if epoch is None:
            iter_string = iter_string.capitalize()
        log_strings.append(iter_string)
    if 'metadata' in result_dict:
        log_strings += result_dict['metadata']
    for key, value in result_dict.items():
        if key != 'metadata':
            format_string = '{}: {:' + get_print_format(value) + '}'
            log_strings.append(format_string.format(key, value))
    if lr is not None:
        log_strings.append('lr: {:.3e}'.format(lr))
    if timer is not None:
        log_strings.append(timer.tostring())
    message = ', '.join(log_strings)
    return message


def get_print_format(value):
    if isinstance(value, int):
        return 'd'
    if isinstance(value, str):
        return 's'
    if value == 0:
        return '.3f'
    if value < 1e-6:
        return '.3e'
    if value < 1e-3:
        return '.6f'
    return '.3f'


def random_sample_rotation(rotation_factor: float = 1.0) -> np.ndarray:
    # angle_z, angle_y, angle_x
    euler = np.random.rand(3) * np.pi * 2 / rotation_factor  # (0, 2 * pi / rotation_range)
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    return rotation


def random_sample_rotation_v2() -> np.ndarray:
    axis = np.random.rand(3) - 0.5
    axis = axis / np.linalg.norm(axis) + 1e-8
    theta = np.pi * np.random.rand()
    euler = axis * theta
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    return rotation


def main():
    path = './abc/'
    dir_exist(path)


if __name__ == '__main__':
    main()
