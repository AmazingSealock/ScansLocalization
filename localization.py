#融合了拼接和定位

import open3d as o3d
import numpy as np
import time

def create_transformation_matrix(translation, rotation_degrees):
    # 将角度转换为弧度
    rotation_radians = np.deg2rad(rotation_degrees)
    
    # 创建旋转矩阵（使用ZYX顺序，即先绕Z轴旋转，再绕Y轴，最后绕X轴）
    R = o3d.geometry.get_rotation_matrix_from_xyz(rotation_radians)
    
    # 创建齐次变换矩阵
    transformation_matrix = np.eye(4)  # 创建4x4单位矩阵
    transformation_matrix[:3, :3] = R  # 设置旋转部分
    transformation_matrix[:3, 3] = translation  # 设置平移部分
    
    return transformation_matrix


# 设置文件路径
# 读取第一个ply文件
ply_file1 = o3d.io.read_point_cloud("./sourcePLY/up.ply")

# 读取第二个ply文件
ply_file2 = o3d.io.read_point_cloud("./sourcePLY/back.ply")

# 读取第三个ply文件
ply_file3 = o3d.io.read_point_cloud("./sourcePLY/front.ply")

# 读取第四个ply文件
ply_file4 = o3d.io.read_point_cloud("./sourcePLY/right.ply")

# 读取第五个ply文件
ply_file5 = o3d.io.read_point_cloud("./sourcePLY/left.ply")

# 加载点云数据
# ply_file1.paint_uniform_color([1.0, 0.0, 0.0])  # RGB值，红色
# ply_file2.paint_uniform_color([0.0, 1.0, 0.0])  # RGB值，绿色
# ply_file3.paint_uniform_color([0.0, 0.0, 1.0])  # RGB值，红色
# ply_file4.paint_uniform_color([1.0, 1.0, 0.0])  # RGB值，绿色
# ply_file5.paint_uniform_color([1.0, 0.0, 1.0])  # RGB值，红色


# 初步查看两个点云
# o3d.visualization.draw_geometries([ply_file1, ply_file3])

# 手眼标定矩阵 机械臂末端->相机坐标系
translation_camera = np.array([
    [-0.993858,  -0.0157531, -0.10954,   0.0562494],
    [ 0.0150722, -0.999862,   0.00704117, 0.0639663],
    [-0.109636,   0.00534691, 0.993957,   0.0957409],
    [0,           0,          0,          1]
])

# 我们需要它的逆矩阵，相机坐标系中的点移动到机械臂坐标系
# camera_to_end_effector = np.linalg.inv(translation_camera)

# 第一个点云的机械臂末端位置
translation1 = [0, 0, 0]
rotation_degrees1 = [0, 0, 0]
transformation_matrix1 = create_transformation_matrix(translation1, rotation_degrees1)
# print("transformation_matrix1:\n", transformation_matrix1)

# 第二个点云的机械臂末端
translation2 = [0, -0.18, 0.00]
rotation_degrees2 = [-20, 0, 0]
# translation2 = [0, -0.145, 0.018]
# rotation_degrees2 = [-20, 0, 0]
transformation_matrix2 = create_transformation_matrix(translation2, rotation_degrees2)
# print("transformation_matrix2:\n", transformation_matrix2)

# 第三个点云的机械臂末端
translation3 = [0, 0.18, 0.00]
rotation_degrees3 = [20, 0, 0]
transformation_matrix3 = create_transformation_matrix(translation3, rotation_degrees3)
# print("transformation_matrix3:\n", transformation_matrix3)

# 第四个点云的机械臂末端
translation4 = [0.18, 0, 0.00]
rotation_degrees4 = [0, -20, 0]
transformation_matrix4 = create_transformation_matrix(translation4, rotation_degrees4)
# print("transformation_matrix5:\n", transformation_matrix5)

# 第五个点云的机械臂末端
translation5 = [-0.18, 0, 0.00]
rotation_degrees5 = [0, 20, 0]
transformation_matrix5 = create_transformation_matrix(translation5, rotation_degrees5)
# print("transformation_matrix5:\n", transformation_matrix5)

# 开始计时
start_time = time.time()

# 先将机械臂的位置转换到相机坐标系下
transformation_matrix1_camera = translation_camera @ transformation_matrix1
transformation_matrix2_camera = translation_camera @ transformation_matrix2
transformation_matrix3_camera = translation_camera @ transformation_matrix3
transformation_matrix4_camera = translation_camera @ transformation_matrix4
transformation_matrix5_camera = translation_camera @ transformation_matrix5

# 计算第二个点云相对于第一个点云的相对变换矩阵
relative_transformation2 = np.linalg.inv(transformation_matrix1_camera) @ transformation_matrix2_camera

relative_transformation3 = np.linalg.inv(transformation_matrix1_camera) @ transformation_matrix3_camera

relative_transformation4 = np.linalg.inv(transformation_matrix1_camera) @ transformation_matrix4_camera

relative_transformation5 = np.linalg.inv(transformation_matrix1_camera) @ transformation_matrix5_camera

# 点云应用相对变换
ply_file2.transform(relative_transformation2)
ply_file3.transform(relative_transformation3)
ply_file4.transform(relative_transformation4)
ply_file5.transform(relative_transformation5)

#------------------------------------------------------------------------------------------------------
# 法线估计
ply_file1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# 使用平面分割去除桌面
plane_model1, inliers1 = ply_file1.segment_plane(distance_threshold=0.005,
                                              ransac_n=3,
                                              num_iterations=1000)
[a, b, c, d] = plane_model1
# print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# 提取剩余点云（田字格部分）
outlier_cloud1 = ply_file1.select_by_index(inliers1, invert=True)

# 统计滤波去除离群点
cl1, ind1 = outlier_cloud1.remove_statistical_outlier(nb_neighbors=80, std_ratio=0.5)

# 保留滤波后的点云
pcd_filtered1 = outlier_cloud1.select_by_index(ind1)



# 法线估计
ply_file2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# 使用平面分割去除桌面
plane_model2, inliers2 = ply_file2.segment_plane(distance_threshold=0.005,
                                              ransac_n=3,
                                              num_iterations=1000)
[a, b, c, d] = plane_model2
# print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# 提取剩余点云（田字格部分）
outlier_cloud2 = ply_file2.select_by_index(inliers2, invert=True)

# 统计滤波去除离群点
cl2, ind2 = outlier_cloud2.remove_statistical_outlier(nb_neighbors=40, std_ratio=0.5)

# 保留滤波后的点云
pcd_filtered2 = outlier_cloud2.select_by_index(ind2)



# 法线估计
ply_file3.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# 使用平面分割去除桌面
plane_model3, inliers3 = ply_file3.segment_plane(distance_threshold=0.005,
                                              ransac_n=3,
                                              num_iterations=1000)
[a, b, c, d] = plane_model3
# print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# 提取剩余点云（田字格部分）
outlier_cloud3 = ply_file3.select_by_index(inliers3, invert=True)

# 统计滤波去除离群点
cl3, ind3 = outlier_cloud3.remove_statistical_outlier(nb_neighbors=40, std_ratio=0.5)

# 保留滤波后的点云
pcd_filtered3 = outlier_cloud3.select_by_index(ind3)



# 法线估计
ply_file4.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# 使用平面分割去除桌面
plane_model4, inliers4 = ply_file4.segment_plane(distance_threshold=0.005,
                                              ransac_n=3,
                                              num_iterations=1000)
[a, b, c, d] = plane_model4
# print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# 提取剩余点云（田字格部分）
outlier_cloud4 = ply_file4.select_by_index(inliers4, invert=True)

# 统计滤波去除离群点
cl4, ind4 = outlier_cloud4.remove_statistical_outlier(nb_neighbors=40, std_ratio=0.5)

# 保留滤波后的点云
pcd_filtered4 = outlier_cloud4.select_by_index(ind4)


# 法线估计
ply_file5.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# 使用平面分割去除桌面
plane_model5, inliers5 = ply_file5.segment_plane(distance_threshold=0.005,
                                              ransac_n=3,
                                              num_iterations=1000)
[a, b, c, d] = plane_model5
# print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# 提取剩余点云（田字格部分）
outlier_cloud5 = ply_file5.select_by_index(inliers5, invert=True)

# 统计滤波去除离群点
cl5, ind5 = outlier_cloud5.remove_statistical_outlier(nb_neighbors=40, std_ratio=0.5)

# 保留滤波后的点云
pcd_filtered5 = outlier_cloud5.select_by_index(ind5)




# 查看去掉平面的点云
# o3d.visualization.draw_geometries([pcd_filtered1, pcd_filtered2, pcd_filtered3, pcd_filtered4, pcd_filtered5])





#------------------------------------------------------------------------------------------------------

# 进行配准
# 估计法线
pcd_filtered1.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
pcd_filtered2.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
pcd_filtered3.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
pcd_filtered4.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
pcd_filtered5.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# 下采样
voxel_size = 0.01
pcd_filtered1_down = pcd_filtered1.voxel_down_sample(voxel_size)
pcd_filtered2_down = pcd_filtered2.voxel_down_sample(voxel_size)
pcd_filtered3_down = pcd_filtered3.voxel_down_sample(voxel_size)
pcd_filtered4_down = pcd_filtered4.voxel_down_sample(voxel_size)
pcd_filtered5_down = pcd_filtered5.voxel_down_sample(voxel_size)

# 计算 FPFH 特征
pcd1_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    pcd_filtered1_down,
    o3d.geometry.KDTreeSearchParamHybrid(radius=10 * voxel_size, max_nn=100))
pcd2_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    pcd_filtered2_down,
    o3d.geometry.KDTreeSearchParamHybrid(radius=10 * voxel_size, max_nn=100))
pcd3_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    pcd_filtered3_down,
    o3d.geometry.KDTreeSearchParamHybrid(radius=10 * voxel_size, max_nn=100))
pcd4_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    pcd_filtered4_down,
    o3d.geometry.KDTreeSearchParamHybrid(radius=10 * voxel_size, max_nn=100))
pcd5_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    pcd_filtered5_down,
    o3d.geometry.KDTreeSearchParamHybrid(radius=10 * voxel_size, max_nn=100))

# 使用 Fast Global Registration
reg_fgr2 = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
    pcd_filtered2_down, pcd_filtered1_down, pcd2_fpfh, pcd1_fpfh,
    o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=voxel_size * 2, iteration_number=64))

reg_fgr3 = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
    pcd_filtered3_down, pcd_filtered1_down, pcd3_fpfh, pcd1_fpfh,
    o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=voxel_size * 2, iteration_number=64))

reg_fgr4 = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
    pcd_filtered4_down, pcd_filtered1_down, pcd4_fpfh, pcd1_fpfh,
    o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=voxel_size * 2, iteration_number=64))

reg_fgr5 = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
    pcd_filtered5_down, pcd_filtered1_down, pcd5_fpfh, pcd1_fpfh,
    o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=voxel_size * 2, iteration_number=64))

# print(f"Transformation:\n {reg_fgr3.transformation}")

pcd_filtered2.transform(reg_fgr2.transformation)
pcd_filtered3.transform(reg_fgr3.transformation)
pcd_filtered4.transform(reg_fgr4.transformation)
pcd_filtered5.transform(reg_fgr5.transformation)
# o3d.visualization.draw_geometries([pcd_filtered1, pcd_filtered2, pcd_filtered3, pcd_filtered4, pcd_filtered5])

threshold = 0.013  # 根据配准精度需求，设置较小的阈值(目前参数是根据手眼标定的误差分析确定的)
init_transformation = np.eye(4)

reg_p2p2 = o3d.pipelines.registration.registration_generalized_icp(
    pcd_filtered2, pcd_filtered1, threshold, init_transformation,
    o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))

reg_p2p3 = o3d.pipelines.registration.registration_generalized_icp(
    pcd_filtered3, pcd_filtered1, threshold, init_transformation,
    o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))

reg_p2p4 = o3d.pipelines.registration.registration_generalized_icp(
    pcd_filtered4, pcd_filtered1, threshold, init_transformation,
    o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))

reg_p2p5 = o3d.pipelines.registration.registration_generalized_icp(
    pcd_filtered5, pcd_filtered1, threshold, init_transformation,
    o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))

# print(f"Transformation after ICP:\n {reg_p2p3.transformation}")
# 应用变换
pcd_filtered2.transform(reg_p2p2.transformation)
pcd_filtered3.transform(reg_p2p3.transformation)
pcd_filtered4.transform(reg_p2p4.transformation)
pcd_filtered5.transform(reg_p2p5.transformation)

# o3d.visualization.draw_geometries([pcd_filtered1, pcd_filtered2, pcd_filtered3, pcd_filtered4, pcd_filtered5])

ply_file2.transform(reg_fgr2.transformation @ reg_p2p2.transformation)
ply_file3.transform(reg_fgr3.transformation @ reg_p2p3.transformation)
ply_file4.transform(reg_fgr4.transformation @ reg_p2p4.transformation)
ply_file5.transform(reg_fgr5.transformation @ reg_p2p5.transformation)

# o3d.visualization.draw_geometries([ply_file2, ply_file3, ply_file3, ply_file4, ply_file5])

combined_ply = ply_file1 + ply_file2 + ply_file3 + ply_file4 + ply_file5

# 下采样点云
voxel_size = 0.0005  # 调整体素大小进行加速
target_points = combined_ply.voxel_down_sample(voxel_size)

# 显示合成后的点云
o3d.visualization.draw_geometries([target_points])

# 将处理后的文件存储下来
# o3d.io.write_point_cloud("./sourcePLY/allpoint.ply", target_points, write_ascii=True)
# print("write")


#------------------------------------------------------------------------------------------------------
#开始定位

# 设置文件路径
source_file_path = "./targetPLY/target1.ply"  # 语义点云地图

# 加载点云数据
source_points = o3d.io.read_point_cloud(source_file_path)
source_points.paint_uniform_color([1.0, 0.0, 0.0])  # RGB值，红色
target_points.paint_uniform_color([0.0, 1.0, 0.0])  # RGB值，绿色

#上下翻转
rotation_matrix = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, -1]])

target_points.rotate(rotation_matrix)

# 初步查看两个点云
# o3d.visualization.draw_geometries([source_points, target_points])

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

threshold = 0.01  
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