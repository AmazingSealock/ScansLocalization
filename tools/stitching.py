#手动调整点云拼接算法

import open3d as o3d
import numpy as np

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

color1 = [1, 0, 0]  # Red for the first point cloud
color2 = [0, 1, 0]  # Green for the second point cloud
color3 = [0, 0, 1]  # Blue for the first point cloud
color4 = [1, 1, 0]  # Green for the second point cloud
color5 = [1, 0, 1]  # Red for the first point cloud

# 读取第一个ply文件
ply_file1 = o3d.io.read_point_cloud("../sourcePLY/up.ply")

# 读取第二个ply文件
ply_file2 = o3d.io.read_point_cloud("../sourcePLY/back.ply")

# 读取第三个ply文件
ply_file3 = o3d.io.read_point_cloud("../sourcePLY/front.ply")

# 读取第四个ply文件
ply_file4 = o3d.io.read_point_cloud("../sourcePLY/right.ply")

# 读取第五个ply文件
ply_file5 = o3d.io.read_point_cloud("../sourcePLY/left.ply")


#z是高度

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
translation2 = [0, -0.145, 0.018]
rotation_degrees2 = [-20, 0, 0]
transformation_matrix2 = create_transformation_matrix(translation2, rotation_degrees2)
# print("transformation_matrix2:\n", transformation_matrix2)

# 第三个点云的机械臂末端
translation3 = [0, 0.15, -0.03]
rotation_degrees3 = [20, 0, 0]
transformation_matrix3 = create_transformation_matrix(translation3, rotation_degrees3)
# print("transformation_matrix3:\n", transformation_matrix3)

# 第四个点云的机械臂末端
translation4 = [0.16, 0, -0.005]
rotation_degrees4 = [0, -20, 0]
transformation_matrix4 = create_transformation_matrix(translation4, rotation_degrees4)
# print("transformation_matrix5:\n", transformation_matrix5)

# 第五个点云的机械臂末端
translation5 = [-0.155, 0, -0.005]
rotation_degrees5 = [0, 20, 0]
transformation_matrix5 = create_transformation_matrix(translation5, rotation_degrees5)
# print("transformation_matrix5:\n", transformation_matrix5)

# 先将机械臂的位置转换到相机坐标系下
transformation_matrix1_camera = translation_camera @ transformation_matrix1
transformation_matrix2_camera = translation_camera @ transformation_matrix2
transformation_matrix3_camera = translation_camera @ transformation_matrix3
transformation_matrix4_camera = translation_camera @ transformation_matrix4
transformation_matrix5_camera = translation_camera @ transformation_matrix5
# print("translation_camera:\n", translation_camera)
# print("transformation_matrix1_camera:\n", transformation_matrix1_camera)
# print("transformation_matrix2_camera:\n", transformation_matrix2_camera)

# 计算第二个点云相对于第一个点云的相对变换矩阵
relative_transformation2 = np.linalg.inv(transformation_matrix1_camera) @ transformation_matrix2_camera

# 计算第二个点云相对于第一个点云的相对变换矩阵
relative_transformation3 = np.linalg.inv(transformation_matrix1_camera) @ transformation_matrix3_camera

# 计算第二个点云相对于第一个点云的相对变换矩阵
relative_transformation4 = np.linalg.inv(transformation_matrix1_camera) @ transformation_matrix4_camera

# 计算第二个点云相对于第一个点云的相对变换矩阵
relative_transformation5 = np.linalg.inv(transformation_matrix1_camera) @ transformation_matrix5_camera

# print("relative_transformation:\n", relative_transformation)

# 点云应用相对变换
ply_file2.transform(relative_transformation2)
ply_file3.transform(relative_transformation3)
ply_file4.transform(relative_transformation4)
ply_file5.transform(relative_transformation5)

# 合并两个点云
combined_ply = ply_file1 + ply_file2 + ply_file3 + ply_file4 + ply_file5
# combined_ply = ply_file1 + ply_file2

# 下采样点云
voxel_size = 0.001  # 调整体素大小进行加速
source_down = combined_ply.voxel_down_sample(voxel_size)

# 显示合成后的点云
o3d.visualization.draw_geometries([source_down])

# 将处理后的文件存储下来
o3d.io.write_point_cloud("./sourcePLY/allpoints.ply", source_down, write_ascii=True)
print("write")