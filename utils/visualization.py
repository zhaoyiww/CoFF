# ---------------------------------------------------------------------------- #
# visualize point clouds and images
# ---------------------------------------------------------------------------- #
import open3d as o3d
import copy as copy


def draw_registration_result_multiple(src, tgt, transformation, src_keypts, window_name):
    source_temp = copy.deepcopy(src)
    target_temp = copy.deepcopy(tgt)
    source_temp.transform(transformation)

    src_keypts_temp = copy.deepcopy(src_keypts)
    src_keypts_temp.paint_uniform_color([1, 0, 0])
    # src_keypts_temp.transform(transformation)

    source_temp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    target_temp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    # custom colors: [0, 0.839, 1.] ; [0.921, 0.569, 0]
    # default colors: [1, 0.706, 0] ; [0, 0.651, 0.929]
    source_temp.paint_uniform_color([0.921, 0.569, 0])
    target_temp.paint_uniform_color([0, 0.839, 1.])

    o3d.visualization.draw_geometries([source_temp, target_temp, src_keypts_temp], window_name=window_name)


def draw_registration_result_single(src_keypts, color=[1,0,0], window_name='open3d'):
    src_keypts_temp = copy.deepcopy(src_keypts)
    src_keypts_temp.paint_uniform_color(color)

    src_keypts_temp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.9, max_nn=30))

    o3d.visualization.draw_geometries([src_keypts_temp], window_name=window_name)


def draw_registration_result_single_pcd_with_keypts(src_pcd, src_keypts, color=[1,0,0], window_name='open3d'):
    source_temp = copy.deepcopy(src_pcd)
    src_keypts_temp = copy.deepcopy(src_keypts)

    source_temp.paint_uniform_color([0.921, 0.569, 0])
    src_keypts_temp.paint_uniform_color(color)

    o3d.visualization.draw_geometries([source_temp, keypoints_to_spheres(src_keypts_temp)], window_name=window_name)

def draw_registration_result(source, target, transformation, true_color=True, window_name='Open3D', visualize_keypts=False):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)

    source_temp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.9, max_nn=30))
    if not visualize_keypts:
        target_temp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
        target_temp.paint_uniform_color([0, 0.839, 1.])
    # else:
    #     target_temp.paint_uniform_color([0, 0.651, 0.929])

    # target_temp.estimate_normals()
    if not true_color:
        # yellow and blue
        source_temp.paint_uniform_color([0.921, 0.569, 0])

    o3d.visualization.draw_geometries([source_temp, target_temp], window_name=window_name, height=500, width=1000, left=10, top=10)


# http://www.open3d.org/docs/latest/tutorial/Advanced/iss_keypoint_detector.html
# This function is only used to make the keypoints
# look better on the rendering
def keypoints_to_spheres(keypoints, color_type=0):
    spheres = o3d.geometry.TriangleMesh()
    for keypoint in keypoints.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(keypoint)
        spheres += sphere
    if color_type == 1:
        spheres.paint_uniform_color([0.33, 0.0, 0.67])
    else:
        spheres.paint_uniform_color([1, 0, 0.])
    return spheres


# http://www.open3d.org/docs/latest/tutorial/Advanced/iss_keypoint_detector.html
# This function is only used to make keypoints look better on the rendering
def pts_to_spheres(keypoints, color_type=0):
    spheres = o3d.geometry.TriangleMesh()
    for keypoint in keypoints.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(keypoint)
        spheres += sphere
    if color_type == 1:
        spheres.paint_uniform_color([0.33, 0.0, 0.67])
    else:
        spheres.paint_uniform_color([1, 0, 0.])
    return spheres


def draw_registration_result_three_views(frame_1, frame_2, frame_3, true_color=False, window_name='Open3D', visualize_keypts=False):
    frame_1_temp = copy.deepcopy(frame_1)
    frame_2_temp = copy.deepcopy(frame_2)
    frame_3_temp = copy.deepcopy(frame_3)

    frame_1_temp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    if not visualize_keypts:
        frame_2_temp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
        # blue
        frame_2_temp.paint_uniform_color([1, 0.706, 0])

        frame_3_temp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
        frame_3_temp.paint_uniform_color([0.555, 0.555, 0.555])
    # else:
    #     target_temp.paint_uniform_color([0, 0.651, 0.929])

    # target_temp.estimate_normals()
    if not true_color:
        # yellow and blue
        # yellow
        frame_1_temp.paint_uniform_color([0, 0.651, 0.929])

    o3d.visualization.draw_geometries([frame_1_temp, frame_2_temp, frame_3_temp], window_name=window_name)