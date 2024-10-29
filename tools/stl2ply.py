import open3d as o3d
import numpy as np

# 读入STL文件
mesh = o3d.io.read_triangle_mesh("../targetPLY/tianzi.STL")

# 对Mesh文件进行采样，生成点云
sample_points_num = 500000
pointcloud = mesh.sample_points_poisson_disk(sample_points_num)
print("sample")
 
# 将单位从mm转为m
pointcloud.points = o3d.utility.Vector3dVector(0.001 * np.float32(pointcloud.points))
print("transfer")
 
# 将处理后的文件存储下来
o3d.io.write_point_cloud("../targetPLY/target1.ply", pointcloud, write_ascii=True)
print("write")

# # 可视化点云
o3d.visualization.draw_geometries([pointcloud], window_name="Point Cloud Visualization")