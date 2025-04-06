import os
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

def calculate_distance(pose1, pose2):

    return math.sqrt((pose1[0] - pose2[0]) ** 2 + (pose1[1] - pose2[1]) ** 2 + (pose1[2] - pose2[2]) ** 2)


def calculate_dot_product(quat1, quat2):
    return sum(a * b for a, b in zip(quat1, quat2))

def normalize_quaternion(quat):
    norm = math.sqrt(sum(a * a for a in quat))
    return [a / norm for a in quat]

def calculate_rotation_angle(matrix1, matrix2,i,j):
    try:
        rotation1 = R.from_matrix(matrix1)
        rotation2 = R.from_matrix(matrix2)


        # 计算两个旋转之间的相对旋转
        relative_rotation = rotation2 * rotation1.inv()

        # 获取旋转角度
        angle = relative_rotation.magnitude()

        # 转换为度数
        angle_degrees = np.degrees(angle)

        # 返回旋转角度
        return angle_degrees
    except Exception as e:
        return 100


def read_poses_scan(pose_dir):
    poses = []
    frame_files = [f for f in os.listdir(pose_dir) if f.endswith('.txt')]
    frame_files.sort(key=lambda x: int(x.split('.')[0]))
    for file_name in frame_files:
        file_path = os.path.join(pose_dir, file_name)
        with open(file_path, 'r') as file:
            pose = [float(num) for num in file.read().strip().split()]
            poses.append(pose)
    return poses
def read_poses_replica(pose_dir):
    poses = []
    with open(pose_dir, 'r') as file:
        for line in file:
            if line.strip():
                pose = [float(num) for num in line.strip().split()]
                pose_matrix = np.array(pose).reshape(4, 4)
                poses.append(pose_matrix)
    return poses

def find_loop_closure(pose_dir,output_file,idlist,distance_threshold=1, index_threshold=30
                       , rotation_threshold=35):
    poses = read_poses_scan(pose_dir)

    loop = []
    for i in idlist:
        for j in range(i+5,len(poses)-1,5):

            pose_i = np.array(poses[i]).reshape((4, 4))
            pose_j = np.array(poses[j]).reshape((4, 4))

            distance = calculate_distance(pose_i[:3,3], pose_j[:3,3])
            rotation_angle = calculate_rotation_angle(pose_i[:3,:3], pose_j[:3,:3],i,j)
            if j - i < 10:
                print("i-j - distance - rotation_angle",i,j,distance,rotation_angle)

            if distance < distance_threshold and rotation_angle < rotation_threshold and j-i>index_threshold:

                loop.append([i,j])

    # 将结果写入 TXT 文件
    with open(output_file, "w") as f:
        for item in loop:
            f.write("%s\n" % " ".join(map(str, item)))


    return loop
