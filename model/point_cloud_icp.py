import numpy as np
import scipy.spatial
from scipy.spatial.transform import Rotation as R
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
import open3d as o3d
import cv2
import os
import glob
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors


# def generate_point_cloud(batch,c2w,fx, fy, cx, cy):
def generate_point_cloud(rgb,depth, c2w, fx, fy, cx, cy):
    color_data = rgb.numpy()
    depth_data = depth.numpy()
    c2w = c2w.cpu().numpy()
    depth_shape = depth_data.shape
    if len(depth_shape) == 2:
        H, W = depth_shape
    else:
        _,H, W= depth_data.shape


    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = (x - cx) / fx
    y = (y - cy) / fy

    z = depth_data
    x = x * z
    y = y * z

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    points = np.dot(points, c2w[:3, :3].T) + c2w[:3, 3]

    colors = color_data.reshape(-1, 3)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(colors)

    return pc

def icp(source, target, trans_init):
    threshold = 0.005
    # 将其从 GPU 移到 CPU（如果需要的话）
    init_est = trans_init.cpu()

    # 将张量转换为 NumPy 数组
    init_est_numpy = init_est.numpy()

    # 确保 NumPy 数组的数据类型是 float64
    init_est_numpy = init_est_numpy.astype(np.float64)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_est_numpy,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    edge_tensor = torch.from_numpy(reg_p2p.transformation).double()

    return  edge_tensor,reg_p2p

def visualize_semantic_segmentation(points, preds,colour_code,output_file):
    """
    Visualizes semantic segmentation on a point cloud and saves the visualization as a .ply file.

    Args:
    - points: numpy array, shape (N, 3), containing the 3D coordinates of the points in the point cloud.
    - preds: numpy array, shape (N,), containing the predicted labels for each point.
    - label_to_color_map: dictionary mapping label values to RGB colors normalized between 0 and 1.
    - output_file: string, path to save the visualization as a .ply file.
    """
    # Convert predicted labels to RGB colors
    label_to_color_map = {i: color for i, color in enumerate(colour_code)}


    # Convert predicted labels to RGB colors
    color_array = np.array([label_to_color_map[int(label)] for label in preds])

    colors = color_array / 255.0  # Normalize colors to [0, 1]
    # 假设 'points' 包含点云的三维坐标
    points = np.asarray(points, dtype=np.float64)

    # Create a point cloud object
    semantic_pc = o3d.geometry.PointCloud()
    semantic_pc.points = o3d.utility.Vector3dVector(points)
    semantic_pc.colors = o3d.utility.Vector3dVector(colors)


    # Save point cloud visualization as a .ply file
    o3d.io.write_point_cloud(output_file, semantic_pc)


def plot_semantic_map(semantic_map, img_id, path='sem_map'):
    """
    保存语义图到指定文件夹。

    Args:
        semantic_map: 语义图 (RGB格式)
        img_id: 当前图像的ID，用于生成保存的文件名
        path: 保存路径，默认是 'sem_map' 文件夹
    """
    # 如果文件夹不存在，则创建
    if not os.path.exists(path):
        os.makedirs(path)

    # 定义保存的文件名
    save_path = os.path.join(path, f"semantic_map_{img_id}.png")

    # 保存语义图像
    plt.imsave(save_path, semantic_map)
    print(f"Semantic map for image {img_id} saved at {save_path}")

def apply_nyu40_color_map(sem_logits,height,width):
    """
    将语义标签映射为NYU40定义的颜色。
    sem_logits: 语义网络输出的标签
    """
    # # sem_logits = sem_logits.cpu().numpy()  # 转换为numpy格式
    # height, width = sem_logits.shape
    color_map = np.zeros((height, width, 3), dtype=np.uint8)  # 创建一个空的RGB图像

    # 遍历每个语义标签，使用nyu40_colour_code为每个类别分配颜色
    for label in range(len(nyu40_colour_code)):
        mask = (sem_logits == label)
        color_map[mask] = nyu40_colour_code[label]

    return color_map


nyu13_colour_code = (np.array([[0, 0, 0],
                       [0, 0, 1], # BED
                       [0.9137,0.3490,0.1882], #BOOKS
                       [0, 0.8549, 0], #CEILING
                       [0.5843,0,0.9412], #CHAIR
                       [0.8706,0.9451,0.0941], #FLOOR
                       [1.0000,0.8078,0.8078], #FURNITURE
                       [0,0.8784,0.8980], #OBJECTS
                       [0.4157,0.5333,0.8000], #PAINTING
                       [0.4588,0.1137,0.1608], #SOFA
                       [0.9412,0.1373,0.9216], #TABLE
                       [0,0.6549,0.6118], #TV
                       [0.9765,0.5451,0], #WALL
                       [0.8824,0.8980,0.7608]])*255).astype(np.uint8)


# color palette for nyu34 labels
nyu34_colour_code = np.array([
       (0, 0, 0),

       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair

       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
    #    (148, 103, 189),		# bookshelf

       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),       # blinds
       (247, 182, 210),		# desk
       (66, 188, 102),      # shelves

       (219, 219, 141),		# curtain
    #    (140, 57, 197),    # dresser
       (202, 185, 52),      # pillow
    #    (51, 176, 203),    # mirror
       (200, 54, 131),      # floor

       (92, 193, 61),       # clothes
       (78, 71, 183),       # ceiling
       (172, 114, 82),      # books
       (255, 127, 14), 		# refrigerator
       (91, 163, 138),      # tv

       (153, 98, 156),      # paper
       (140, 153, 101),     # towel
    #    (158, 218, 229),		# shower curtain
       (100, 125, 154),     # box
    #    (178, 127, 135),       # white board

    #    (120, 185, 128),       # person
       (146, 111, 194),     # night stand
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209),      # lamp

       (227, 119, 194),		# bathtub
       (213, 92, 176),      # bag
       (94, 106, 211),      # other struct
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)       # other prop
    ]).astype(np.uint8)



# color palette for nyu40 labels
nyu40_colour_code = np.array([
       (0, 0, 0),

       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair

       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf

       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),       # blinds
       (247, 182, 210),		# desk
       (66, 188, 102),      # shelves

       (219, 219, 141),		# curtain
       (140, 57, 197),    # dresser
       (202, 185, 52),      # pillow
       (51, 176, 203),    # mirror
       (200, 54, 131),      # floor

       (92, 193, 61),       # clothes
       (78, 71, 183),       # ceiling
       (172, 114, 82),      # books
       (255, 127, 14), 		# refrigerator
       (91, 163, 138),      # tv

       (153, 98, 156),      # paper
       (140, 153, 101),     # towel
       (158, 218, 229),		# shower curtain
       (100, 125, 154),     # box
       (178, 127, 135),       # white board

       (120, 185, 128),       # person
       (146, 111, 194),     # night stand
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209),      # lamp

       (227, 119, 194),		# bathtub
       (213, 92, 176),      # bag
       (94, 106, 211),      # other struct
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)       # other prop
    ]).astype(np.uint8)

