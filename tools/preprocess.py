#降采样，去掉平面

import open3d as o3d
import numpy as np
import cv2

# 加载点云
pcd = o3d.io.read_point_cloud("../sourcePLY/qinxie1.ply")  # 替换为你的点云文件路径

# 可视化原始点云
# o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")

# 点云下采样（可选）
# pcd_down = pcd.voxel_down_sample(voxel_size=0.00)

# 法线估计
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# 使用平面分割去除桌面
plane_model, inliers = pcd.segment_plane(distance_threshold=0.005,
                                              ransac_n=3,
                                              num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# 提取平面内点云（桌面）
inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])  # 将桌面点云涂成红色

# 提取剩余点云（田字格部分）
outlier_cloud = pcd.select_by_index(inliers, invert=True)

# 统计滤波去除离群点
cl, ind = outlier_cloud.remove_statistical_outlier(nb_neighbors=40, std_ratio=0.25)

# 保留滤波后的点云
pcd_filtered = outlier_cloud.select_by_index(ind)

# 可视化去除桌面后的点云
o3d.visualization.draw_geometries([pcd_filtered], window_name="Filtered Point Cloud")

# 保存提取后的点云
o3d.io.write_point_cloud("./sourcePLY/qinxie2.ply", pcd_filtered, write_ascii=True)
