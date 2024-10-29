#去掉零点，转换单位

import open3d as o3d
import numpy as np

ply_file = "../sourcePLY/up.ply"  # 替换为你的PLY文件路径
pointcloud = o3d.io.read_point_cloud(ply_file)
 
# 将单位从mm转为m
for point in pointcloud.points:
    point[0] *= 0.001
    point[1] *= 0.001
    point[2] *= 0.001

print("transfer")

# 过滤掉坐标为(0, 0, 0)的点
valid_points = []
for point in pointcloud.points:
    if point[0] != 0 or point[1] != 0 or point[2] != 0:
        valid_points.append(point)

pointcloud.points = o3d.utility.Vector3dVector(valid_points)

print("remove zero points")

# 将处理后的文件存储下来
o3d.io.write_point_cloud("../sourcePLY/up.ply", pointcloud, write_ascii=True)
print("write")

# # 可视化点云
o3d.visualization.draw_geometries([pointcloud], window_name="Point Cloud Visualization")