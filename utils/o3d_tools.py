# ---------------------------------------------------------------------------- #
# This script contains functions regarding using the open-source library [Open3D](https://www.open3d.org/)
# ---------------------------------------------------------------------------- #
import numpy as np
import open3d as o3d
import copy
import torch
import matplotlib.pyplot as plt


def icp_registration(src_pcd, tgt_pcd, initial_transform, threshold=0.1, icp_type='point2point'):
    """
    Implement point cloud registration using point2plane_icp
    :param icp_type:
    :param src_pcd:
    :param tgt_pcd:
    :param initial_transform:
    :param threshold: max_correspondence_distance (float) – Maximum correspondence points-pair distance
    :param criteria: optional, default=ICPConvergenceCriteria class with
    relative_fitness=1.000000e-06, relative_rmse=1.000000e-06, and max_iteration=30)

    :return:
    fitness, which measures the overlapping area (# of inlier correspondences / # of points in target).
    The higher the better. inlier_rmse, which measures the RMSE of all inlier correspondences. The lower the better.
    """

    # Compute normal vectors for planes
    src_pcd.estimate_normals()
    tgt_pcd.estimate_normals()

    # call registration_icp in Open3D
    if icp_type == 'point2point':
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
    # estimation_method = (TransformationEstimationPointToPoint, TransformationEstimationPointToPlane,
    # TransformationEstimationForGeneralizedICP, TransformationEstimationForColoredICP)
    # criteria
    elif icp_type == 'point2plane':
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    elif icp_type == 'generalized_icp':
        estimation_method = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(False)
    else:
        raise ValueError('ICP type not supported')

    if icp_type in ['point2point', 'point2plane']:
        result_icp = o3d.pipelines.registration.registration_icp(
            source=src_pcd, target=tgt_pcd, max_correspondence_distance=threshold, init=initial_transform,
            estimation_method=estimation_method,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria
            (relative_fitness=1e-06, relative_rmse=1e-06, max_iteration=30))
    elif icp_type == 'generalized_icp':
        result_icp = o3d.pipelines.registration.registration_generalized_icp(
            source=src_pcd, target=tgt_pcd, max_correspondence_distance=threshold, init=initial_transform,
            estimation_method=estimation_method,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria
            (relative_fitness=1e-06, relative_rmse=1e-06, max_iteration=30))
    else:
        raise ValueError('ICP type not supported')

    src_corr_pts, tgt_corr_pts = get_correspondence_pairwise_point_clouds(src_pcd, tgt_pcd,
                                                                          result_icp.correspondence_set)
    result_dict_icp = {
        "fitness": result_icp.fitness,
        "inlier_rmse": result_icp.inlier_rmse,
        "correspondence_set": np.asarray(result_icp.correspondence_set),
        "est_transform": result_icp.transformation,
        "src_corr_pts": src_corr_pts,
        "tgt_corr_pts": tgt_corr_pts
    }

    return result_dict_icp


def colored_icp(source, target, initial_transform, voxel_radius=None, max_iter=None):
    """
    Implement point cloud registration using colored_icp
    :param source:
    :param target:
    :param initial_transform:
    :param voxel_radius:
    :param max_iter:
    :return:
    """
    # multi_scale colored_icp\
    if voxel_radius is None:
        voxel_radius = [0.04, 0.02, 0.01]
    if max_iter is None:
        max_iter = [50, 30, 14]
    current_transformation = np.identity(4)
    # print("====================================================================")
    # print("Colored point cloud registration")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        # print([iter, radius, scale])

        # print("(1). Downsample with a voxel size %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        # print("(2). Estimate normal.")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        # print("(3). Applying colored point cloud registration")
        result_colored_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=iter))
        current_transformation = result_colored_icp.transformation
        # print(result_colored_icp)
    # print("====================================================================")
    # draw_registration_result(source, target, result_colored_icp.transformation, window_name="colored ICP result")
    src_corr_pts, tgt_corr_pts = get_correspondence_pairwise_point_clouds(source_down, target_down,
                                                                          result_colored_icp.correspondence_set)
    result_dict_colored_icp = {
        "fitness": result_colored_icp.fitness,
        "inlier_rmse": result_colored_icp.inlier_rmse,
        "correspondence_set": np.asarray(result_colored_icp.correspondence_set),
        "est_transform": result_colored_icp.transformation,
        "src_corr_pts": src_corr_pts,
        "tgt_corr_pts": tgt_corr_pts
    }
    return result_dict_colored_icp


def get_correspondence_pairwise_point_clouds(src_key_pcd, tgt_key_pcd, corr_indices_set):
    """
    Get the correspondence points
    :param src_key_pcd:
    :param tgt_key_pcd:
    :param corr_indices_set:
    :return:
    """
    src_key_pts = pcd2array(src_key_pcd, return_colors=False)
    tgt_key_pts = pcd2array(tgt_key_pcd, return_colors=False)
    corr_indices = np.asarray(corr_indices_set)
    # get the correspondence points
    src_corr_pts = src_key_pts[corr_indices[:, 0], :]
    tgt_corr_pts = tgt_key_pts[corr_indices[:, 1], :]
    return src_corr_pts, tgt_corr_pts


def ransac_registration(src_pts, tgt_pts, corres, distance_threshold=0.05, ransac_n=3,
                        max_iteration=50000, max_validation=1000, ransac_type='point2point'):
    """
    Implement point cloud registration using RANSAC
    :param ransac_type:
    :param src_pts:
    :param tgt_pts:
    :param corres:
    :param max_corres_dist:
    :param voxel_size:
    :param ransac_n:
    :param max_iteration:
    :param max_validation:
    :return:
    """
    if ransac_type == 'point2point':
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
    elif ransac_type == 'point2plane':
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane(False)
    else:
        raise ValueError('RANSAC type not supported')
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source=src_pts, target=tgt_pts, corres=corres,
        # checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_dist)],
        max_correspondence_distance=distance_threshold,
        estimation_method=estimation_method,
        ransac_n=ransac_n,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration, max_validation)
    )
    return result_ransac


def array2pcd(point_array, colors=None, normals=None):
    """
    Convert an array to a point cloud
    :param point_array:
    :param colors: true colors if available, default None
    :param normals: estimated normals if available, default None
    :return: open3d.geometry.PointCloud
    """

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_array)
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        point_cloud.normals = o3d.utility.Vector3dVector(normals)
    return point_cloud


def tensor2pcd(point_tensor, colors=None):
    """
    Convert a tensor to a point cloud
    :param point_array:
    :param colors: true colors if available, default None
    :return: open3d.geometry.PointCloud
    """
    point_array = array2tensor(point_tensor, invert=True)
    if colors is not None:
        if isinstance(colors, torch.Tensor):
            colors = array2tensor(colors, invert=True)
    point_cloud = array2pcd(point_array, colors=colors)
    return point_cloud


def pcd2array(point_cloud, return_colors=False):
    """
    Convert a point cloud to an array
    :param point_cloud: open3d.geometry.PointCloud
    :param color: true colors if available, default None
    :return: numpy.ndarray with shape (n_points, 3)
    """

    src_pcd = copy.deepcopy(point_cloud)
    src_points = np.asarray(src_pcd.points)
    if return_colors is True:
        src_colors = np.asarray(src_pcd.colors)
        return src_points, src_colors
    else:
        return src_points


def array2tensor(array, invert=False):
    if invert:
        return np.asarray(array.cpu())
    else:
        if array.dtype == np.uint64:
            array = array.astype(np.int64)
        #     return torch.from_numpy(array)
        # else:
        #     return torch.from_numpy(array).float()
        return torch.from_numpy(array)


def pcd2tensor(point_cloud, device='cuda', return_colors=False):
    """
    Convert a point cloud to an array
    :param point_cloud: open3d.geometry.PointCloud
    :param color: true colors if available, default None
    :return: numpy.ndarray with shape (n_points, 3)
    """

    src_pcd = copy.deepcopy(point_cloud)
    src_points = np.asarray(src_pcd.points)
    src_points = torch.from_numpy(src_points).float()
    if return_colors is True:
        src_colors = np.asarray(src_pcd.colors)
        src_colors = torch.from_numpy(src_colors)
        return src_points.to(device), src_colors.to(device)
    else:
        return src_points.to(device)

def visualize_raw_point_clouds(src_pcd, tgt_pcd, offset=[0,0,0], window_name="Open3D"):
    src_pts = pcd2array(src_pcd)
    src_colors = src_pcd.colors
    # tgt_pts = pcd2array(tgt_pcd.points)
    # tgt_colors = tgt_pcd.colors

    src_pts += offset

    src_pcd_temp = array2pcd(src_pts, colors=src_colors)
    tgt_pcd_temp = copy.deepcopy(tgt_pcd)

    src_pcd_temp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
    # src_pcd_temp.paint_uniform_color([0.921, 0.569, 0])
    tgt_pcd_temp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
    # tgt_pcd_temp.paint_uniform_color([0, 0.839, 1.])

    o3d.visualization.draw_geometries([src_pcd_temp, tgt_pcd_temp], window_name=window_name)


def visualize_registration_quality(source, target, transformation, offset=[0,0,0], true_color=False, window_name='Open3D'):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    source_pts, source_color = pcd2array(source_temp, return_colors=True)
    source_pts += offset
    source_temp = array2pcd(source_pts, colors=source_color)
    if not true_color:
        source_temp.paint_uniform_color([0.921, 0.569, 0])
        target_temp.paint_uniform_color([0, 0.839, 1.])
    source_temp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
    target_temp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name=window_name)


# source: GeoTrans
def get_nearest_neighbor(
        q_points: np.ndarray,
        s_points: np.ndarray,
        return_index: bool = False,
):
    r"""Compute the nearest neighbor for the query points in support points."""
    from scipy.spatial import cKDTree

    s_tree = cKDTree(s_points)
    distances, indices = s_tree.query(q_points, k=1, workers=-1)
    if return_index:
        return distances, indices
    else:
        return distances


# source: GeoTrans
def compute_overlap_ratio(point_cloud_1, point_cloud_2, transform=None, positive_radius=0.1):
    r"""Compute the overlap of two point clouds."""
    src_pcd_temp = copy.deepcopy(point_cloud_1)
    tgt_pcd_temp = copy.deepcopy(point_cloud_2)
    if transform is not None:
        # src_points = array_transform(src_pcd, transform)
        src_pcd_temp = src_pcd_temp.transform(transform)
    src_pts, tgt_pts = src_pcd_temp.points, tgt_pcd_temp.points
    nn_distances = get_nearest_neighbor(src_pts, tgt_pts)
    overlap = np.mean(nn_distances < positive_radius)
    return overlap


def list2ndarray(image_pose):
    pose_frame_1 = np.array(image_pose)
    pose_frame_1 = [line.strip() for line in pose_frame_1]
    pose_frame_1 = np.array([line.split() for line in pose_frame_1])
    pose_frame_1 = pose_frame_1.astype(np.float32)
    return pose_frame_1


# do transformation for ndarray format point cloud
def array_transform(src_pts, est_transform):
    rotation, translation = est_transform[:3, :3], est_transform[:3, 3]
    src_pts = np.matmul(src_pts, rotation.T) + translation
    return src_pts


def visualize_patch_match(src_pts, tgt_pts, src_patch_pts, tgt_patch_pts, offset=[5, 0, 0],
                          src_color=[0.921, 0.569, 0], tgt_color=[0, 0.839, 1.],
                          window_name="Open3D"):
    """
    visualize a patch match
    :param src_pts:
    :param tgt_pts:
    :param src_patch_pts:
    :param tgt_patch_pts:
    :param offset:
    :param window_name:
    :return:
    """
    # ensure all torch.tensor variables are converted to numpy.array for visualization
    if isinstance(src_pts, torch.Tensor):
        src_pts = array2tensor(src_pts, invert=True)
    if isinstance(tgt_pts, torch.Tensor):
        tgt_pts = array2tensor(tgt_pts, invert=True)
    if isinstance(src_patch_pts, torch.Tensor):
        src_patch_pts = array2tensor(src_patch_pts, invert=True)
    if isinstance(tgt_patch_pts, torch.Tensor):
        tgt_patch_pts = array2tensor(tgt_patch_pts, invert=True)

    if isinstance(tgt_patch_pts, list):
        # get different colors
        colors = plt.cm.tab20.colors
        # filter red, yellow and blue
        filtered_colors = [color for color in colors if
                           color not in [(1.0, 1.0, 0.0), (0.0, 0.0, 1.0)]]
        for i, patch in enumerate(tgt_patch_pts):
            if isinstance(patch, torch.Tensor):
                patch = array2tensor(patch, invert=True)
            tgt_patch_pts_temp = copy.deepcopy(patch)
            if i == 0:
                tgt_patch_pcd_combine = corr_pts2spheres(tgt_patch_pts_temp, radius=0.5, colors=None)
            else:
                tgt_patch_pcd_temp = corr_pts2spheres(tgt_patch_pts_temp, radius=0.5, colors=np.asarray(filtered_colors[i]))
                tgt_patch_pcd_combine += tgt_patch_pcd_temp
    else:
        tgt_patch_pcd_temp = copy.deepcopy(tgt_patch_pts)
        # tgt_patch_pcd_temp += offset
        tgt_patch_pcd_temp = corr_pts2spheres(tgt_patch_pcd_temp, radius=0.5)

        tgt_patch_pcd_combine = tgt_patch_pcd_temp

    if isinstance(src_patch_pts, list):
        # get different colors, tab10, Set1, Set2, Dark2
        # colors = plt.cm.tab10.colors
        colors = plt.cm.Dark2.colors
        # filter red, yellow and blue
        # filtered_colors = [color for color in colors if
        #                    color not in [(1.0, 1.0, 0.0), (0.0, 0.0, 1.0)]]
        filtered_colors = colors[0:]
        for i, patch in enumerate(src_patch_pts):
            if isinstance(patch, torch.Tensor):
                patch = array2tensor(patch, invert=True)
            src_patch_pts_temp = copy.deepcopy(patch)
            src_patch_pts_temp += offset
            if i == 0:
                src_patch_pcd_combine = corr_pts2spheres(src_patch_pts_temp, radius=0.5, colors=None)
            else:
                src_patch_pts_temp = corr_pts2spheres(src_patch_pts_temp, radius=0.5, colors=np.asarray(filtered_colors[i]))
                src_patch_pcd_combine += src_patch_pts_temp
    else:
        src_patch_pts_temp = copy.deepcopy(src_patch_pts)
        src_patch_pts_temp += offset
        src_patch_pcd_temp = corr_pts2spheres(src_patch_pts_temp, radius=0.5)

        src_patch_pcd_combine = src_patch_pcd_temp

    src_pts_temp = copy.deepcopy(src_pts)
    tgt_pts_temp = copy.deepcopy(tgt_pts)
    # src_patch_pts_temp = copy.deepcopy(src_patch_pts)

    # add offset for visualization
    src_pts_temp += offset
    # src_patch_pts_temp += offset

    src_pcd_temp = array2pcd(src_pts_temp, colors=None)
    tgt_pcd_temp = array2pcd(tgt_pts_temp, colors=None)

    src_pcd_temp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
    src_pcd_temp.paint_uniform_color(src_color)
    tgt_pcd_temp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
    tgt_pcd_temp.paint_uniform_color(tgt_color)

    # src_patch_pcd_temp = corr_pts2spheres(src_patch_pts_temp, radius=0.5)

    o3d.visualization.draw_geometries([src_pcd_temp, tgt_pcd_temp, src_patch_pcd_combine, tgt_patch_pcd_combine],
                                      window_name=window_name,
                                      # height=500, width=1000, left=10, top=10,
                                      # # zoom=0.69999999999999996,
                                      # zoom=0.9,
                                      # front=[0.027687275158018637, -0.33396387594011756, -0.94217914663899138],
                                      # lookat=[0.2879103366877438, 0.24373929337403055, 1.3014192634997928],
                                      # up=[0.0019434050756918942, -0.9425207198478105, 0.33414205936139957]
    )


def visualize_correpondence(src_pts, tgt_pts, corr_pts, offset=[5, 0, 0], num_vis_corr=None, window_name="Open3D"):
    """
    Visualize the correspondence points
    :param src_pcd:
    :param src_corr_pts:
    :param tgt_pcd:
    :param tgt_corr_pts:
    :param inlier_indices:
    :param offset:
    :return:
    """
    corr_pts_temp = copy.deepcopy(corr_pts)
    if corr_pts_temp.shape[0] > num_vis_corr:
        corr_vis_idx = np.random.choice(corr_pts_temp.shape[0], size=num_vis_corr, replace=False)
        # select first num_vis_corr for comparison
        # corr_vis_idx = np.arange(corr_pts_temp.shape[0])[200:200+num_vis_corr]
        corr_pts_temp = corr_pts_temp[corr_vis_idx]
    src_pts_temp = copy.deepcopy(src_pts)
    tgt_pts_temp = copy.deepcopy(tgt_pts)
    src_corr_pts_temp = corr_pts_temp[:, :3]
    tgt_corr_pts_temp = corr_pts_temp[:, 3:]

    # add offset for visualization
    src_pts_temp += offset
    src_corr_pts_temp += offset

    src_pcd_temp = array2pcd(src_pts_temp, colors=None)
    tgt_pcd_temp = array2pcd(tgt_pts_temp, colors=None)

    # corr_lines = o3d.geometry.LineSet()
    corr_pts = np.concatenate([src_corr_pts_temp, tgt_corr_pts_temp], axis=0)
    corr_index = np.array([range(src_corr_pts_temp.shape[0]), range(src_corr_pts_temp.shape[0], corr_pts.shape[0])])
    corr_index = corr_index.T
    corr_lines = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corr_pts),
        lines=o3d.utility.Vector2iVector(corr_index)
    )
    # colors = [[1, 0, 0] for i in range(corr_index.shape[0])]

    # colors = [[1, 0, 0] if inlier_indices[i] == 0 else [0, 1, 0] for i in range(corr_index.shape[0])]
    colors = [[0, 1, 0] for i in range(corr_index.shape[0])]
    corr_lines.colors = o3d.utility.Vector3dVector(colors)

    # src_corr_pts_colors = np.ones_like(src_corr_pts_temp) * np.array([[0, 0, 1]])
    # src_corr_pcd = array2pcd(src_corr_pts_temp, colors=src_corr_pts_colors)
    # tgt_corr_pts_colors = np.ones_like(tgt_corr_pts_temp) * np.array([[0, 0, 1]])
    # tgt_corr_pcd = array2pcd(tgt_corr_pts_temp, colors=tgt_corr_pts_colors)

    src_pcd_temp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
    src_pcd_temp.paint_uniform_color([0.921, 0.569, 0])
    tgt_pcd_temp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
    tgt_pcd_temp.paint_uniform_color([0, 0.839, 1.])
    src_corr_pcd = corr_pts2spheres(src_corr_pts_temp, radius=0.1)
    tgt_corr_pcd = corr_pts2spheres(tgt_corr_pts_temp, radius=0.1)

    # + src_corr_pcd + tgt_corr_pcd
    # combined = src_pcd_temp + tgt_pcd_temp
    # o3d.io.write_point_cloud("combined.ply", combined)
    # o3d.io.write_line_set("corr_lines.ply", corr_lines)

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(src_pcd_temp)
    # vis.add_geometry(tgt_pcd_temp)
    # vis.add_geometry(src_corr_pcd)
    # vis.add_geometry(tgt_corr_pcd)
    # vis.add_geometry(corr_lines)
    # ctr = vis.get_view_control()
    # ctr.change_field_of_view(step=25)
    # vis.get_render_option().load_from_json("./renderoption.json")
    # vis.run()

    o3d.visualization.draw_geometries([src_pcd_temp, tgt_pcd_temp, src_corr_pcd, tgt_corr_pcd, corr_lines],
                                      window_name=window_name,
                                      # boundingbox_max=[608.05023193359375, 503.56170654296875, 1535.5299072265625],
                                      # boundingbox_min=[593.6993408203125, 481.0865478515625, 1517.9112548828125],
                                      # field_of_view=60.0,
                                      # height=1080, width=1920, left=50, top=50,
                                      # zoom=0.59999999999999996,
                                      # front=[-0.92085450514935885, -0.18284775129558059, 0.34437433149449337],
                                      # lookat=[550.55380249023438, 490.20396423339844, 1530.5595703125],
                                      # up=[-0.15504634941638196, 0.98211024227482857, 0.10686487520008947]
    )

# http://www.open3d.org/docs/latest/tutorial/Advanced/iss_keypoint_detector.html
# This function is only used to make the keypoints
# look better on the rendering
def corr_pts2spheres(src_corr_pts, radius, colors=None):
    """
    Convert keypoints to spheres for better visualization
    :param keypoints:
    :return:
    """
    src_corr_pcd = array2pcd(src_corr_pts)
    spheres = o3d.geometry.TriangleMesh()
    for keypoint in src_corr_pcd.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(keypoint)
        spheres += sphere
    if colors is None:
        spheres.paint_uniform_color([1.0, 0, 0.0])
    else:
        spheres.paint_uniform_color(colors)
    return spheres



# global (coarse) registration, ransac feature matching-based registration
def ransac_feature_matching_registration(src_pcd, tgt_pcd, src_feat, tgt_feat,
                                         max_distance=10, ransac_n=4, max_iteration=100000,
                                         max_validation=100):
    """
    Args:
        source_points:
        target_points:
        source_features:
        target_features:
        max_correspondence_distance (float): distance_threshold, is equal to voxel_size * 1.5 (in Open3D)
        # estimation_method: TransformationEstimationPointToPoint
        ransac_n: fit ransac with ransac_n correspondences
        # checkers
        max_iteration: (criteria:) RANSACConvergenceCriteria class with max_iteration = 100000 – Convergence criteria
        max_validation: (criteria:) RANSACConvergenceCriteria class with max_validation = 100 – Convergence criteria

    Return:
        o3d.pipelines.registration.RegistrationResult, i.e.,
        correspondence_set, fitness, inlier_rmse, transformation
        :param target_points:
        :param source_points:
        :param voxel_size:
    """
    # max_correspondence_distance = voxel_size * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_pcd, tgt_pcd, src_feat, tgt_feat, True, max_distance,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), ransac_n,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_distance)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration, max_validation)
    )
    return result


