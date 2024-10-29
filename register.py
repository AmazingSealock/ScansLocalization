import open3d as o3d
import numpy as np


# 设置文件路径
target_file_path = "./sourcePLY/allpoint.ply"  # 实时点云
source_file_path = "./targetPLY/target1.ply"  # 语义点云地图

# 加载点云数据
target_points = o3d.io.read_point_cloud(target_file_path)
source_points = o3d.io.read_point_cloud(source_file_path)
source_points.paint_uniform_color([1.0, 0.0, 0.0])  # RGB值，红色
target_points.paint_uniform_color([0.0, 1.0, 0.0])  # RGB值，绿色

rotation_matrix = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, -1]])

target_points.rotate(rotation_matrix)

# 初步查看两个点云
o3d.visualization.draw_geometries([source_points, target_points])

# 法线估计
target_points.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# 使用平面分割去除桌面
plane_model, inliers = target_points.segment_plane(distance_threshold=0.002,
                                              ransac_n=3,
                                              num_iterations=1000)
[a, b, c, d] = plane_model
# print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# 提取剩余点云（田字格部分）
outlier_cloud = target_points.select_by_index(inliers, invert=True)

# 统计滤波去除离群点
cl, ind = outlier_cloud.remove_statistical_outlier(nb_neighbors=40, std_ratio=1)

# 保留滤波后的点云
target_points_filtered = outlier_cloud.select_by_index(ind)

# o3d.visualization.draw_geometries([target_points_filtered])



#------------------------------------------------------------------------------------------------------

# 进行配准
# 估计法线
target_points_filtered.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
source_points.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# 下采样
voxel_size = 0.01
source_down = source_points.voxel_down_sample(voxel_size)
target_down = target_points_filtered.voxel_down_sample(voxel_size)

source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    source_down,
    o3d.geometry.KDTreeSearchParamHybrid(radius=10 * voxel_size, max_nn=100))
target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    target_down,
    o3d.geometry.KDTreeSearchParamHybrid(radius=10 * voxel_size, max_nn=100))

# 使用 Fast Global Registration
reg_fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
    target_down, source_down, target_fpfh, source_fpfh,
    o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=voxel_size * 2, iteration_number=64))

target_points_filtered.transform(reg_fgr.transformation)
target_down.transform(reg_fgr.transformation)

# 将源点云和目标点云（配准后的）转换为 numpy 数组
source_np = np.asarray(source_points.points)
target_np = np.asarray(target_points_filtered.points)

# 使用 KDTree 查找最近邻点对
target_kd_tree = o3d.geometry.KDTreeFlann(target_points_filtered)
distances = []

for point in source_np:
    # 找到每个源点在目标点云中的最近邻
    [_, idx, _] = target_kd_tree.search_knn_vector_3d(point, 1)
    nearest_point = target_np[idx[0]]
    
    # 计算距离并存储
    distances.append(np.linalg.norm(point - nearest_point))

# 计算平均距离差和 RMSE
average_distance = np.mean(distances)
rmse_distance = np.sqrt(np.mean(np.square(distances)))

print(f"Average Distance Difference: {average_distance}")
print(f"RMSE Distance Difference: {rmse_distance}")

# o3d.visualization.draw_geometries([source_points, target_points_filtered])

#------------------------------------------------------------------------------------------------------


threshold = 0.05  
init_transformation = np.eye(4)

reg_p2p = o3d.pipelines.registration.registration_generalized_icp(
    target_points_filtered, source_points, threshold, init_transformation,
    o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
    o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=500, relative_fitness=1e-6, relative_rmse=1e-6))

print(f"Transformation after ICP:\n {reg_p2p.transformation}")
print(f"Fitness:\n {reg_p2p.fitness}")
print(f"RMSE:\n {reg_p2p.inlier_rmse}")
target_points_filtered.transform(reg_p2p.transformation)

o3d.visualization.draw_geometries([source_points, target_points_filtered])
