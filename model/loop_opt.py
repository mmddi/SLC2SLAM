import os
import torch
import argparse
import model.pypose as pp
from torch import nn
import matplotlib.pyplot as plt
import model.pypose.optim.solver as ppos
import model.pypose.optim.strategy as ppost
from model.pypose.optim.scheduler import StopOnPlateau

import numpy as np
from scipy.spatial.transform import Rotation as R



class PoseGraph(nn.Module):
    def __init__(self, nodes,config):
        """
                        初始化PoseGraph。

                        参数:
                        - nodes (torch.Tensor): 初始节点。
                        """

        super().__init__()

        self.nodes = pp.Parameter(nodes)
        self.initial_poses = nodes.clone()
        self.config = config  # 设置config属性
    def forward(self, edges, poses):
        """
                              PoseGraph的前向传播。

                              参数:
                              - edges (torch.Tensor): 图的边。
                              - poses (torch.Tensor): 图的位姿。

                              返回:
                              - torch.Tensor: 前向传播的结果。
                              """


        node1 = self.nodes[edges[..., 0]]
        node2 = self.nodes[edges[..., 1]]


        error = poses.Inv() @ node2 @ node1.Inv()
        return error.Log().tensor()

@torch.no_grad()
def plot_and_save(points, pngname, title='', axlim=None):
    """
                           绘制3D点并保存绘图。

                           参数:
                           - points (torch.Tensor): 递给函数的3D点云数据。这是一个 PyTorch 张量。
                           - pngname (str): 保存绘图的文件名。
                           - title (str): 绘图的标题。
                           - axlim (tuple): 绘图轴的限制。

                           返回:
                           - tuple: 包含绘图的xlim、ylim和zlim的元组。
                           """
    points = points.detach().cpu().numpy()  # 将输入的 PyTorch 张量转换为 NumPy 数组，并从计算设备中分离。
    plt.figure(figsize=(7, 7))
    ax = plt.axes(projection='3d')
    ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 'b')
    plt.title(title)
    if axlim is not None:
        ax.set_xlim(axlim[0])
        ax.set_ylim(axlim[1])
        ax.set_zlim(axlim[2])
    plt.savefig(pngname)
    print('Saving to', pngname)
    return ax.get_xlim(), ax.get_ylim(), ax.get_zlim()


def calculate_distance_sum(points1, points2):
    """
       计算两组点之间的距离和。

       参数:
       - points1 (torch.Tensor): 第一组点的坐标。
       - points2 (torch.Tensor): 第二组点的坐标。

       返回:
       - float: 两组点之间距离的和。
       """
    distance = torch.norm(points1 - points2, dim=1)
    distance_sum = torch.sum(distance)
    return distance_sum.item()



def pypose(config,edge,conid,est,gt,device,name,save_path):


    edges_list = []
    poses_list = []

    node_indices = []
    for row in edge:
        node_indices.append(torch.tensor(np.array(row[0:2], dtype=np.int64)))

        index1 = conid.index(node_indices[-1][0].item())
        index2 = conid.index(node_indices[-1][1].item())
        new_node_indices = torch.tensor([index1, index2])
        edges_list.append(new_node_indices)
        relative_transformation_list = row[2:]  # -1
        relative_transformation_array = np.array(relative_transformation_list)
        transformation_matrix = relative_transformation_array.reshape(4, 4)
        translation_vector = transformation_matrix[:3, -1]
        rotation_matrix = transformation_matrix[:3, :3]
        rotation_matrix_cpu = rotation_matrix
        r3 = R.from_matrix(rotation_matrix_cpu)
        quaternion = r3.as_quat()
        se3_data = np.concatenate([translation_vector, quaternion])
        se3_tensor = torch.tensor(se3_data, dtype=torch.float32)
        se3_tensor = pp.SE3(se3_tensor)
        poses_list.append(se3_tensor)



    edges_list = torch.stack(edges_list).to(device)
    for i in range(len(poses_list)):
        if poses_list[i].device != device:
            poses_list[i] = poses_list[i].to(device)
    poses_list = torch.stack(poses_list).to(torch.float32).to(device)

    translation_vectors = [pose[:3, -1] for pose in gt.values()]
    for i in range(len(translation_vectors)):
        if translation_vectors[i].device != device:
            translation_vectors[i] = translation_vectors[i].to(device)


    nodes_list = []
    node = []
    gt_node = []
    for i in conid:
        node.append(est[i])
        gt_node.append(translation_vectors[i])
    for frame_idx in range(len(node)):
        transformation_matrix = node[frame_idx]
        translation_vector = transformation_matrix[:3, -1]
        rotation_matrix = transformation_matrix[:3, :3]
        rotation_matrix_cpu = rotation_matrix.cpu().detach().numpy()
        r3 = R.from_matrix(rotation_matrix_cpu)
        quaternion = r3.as_quat()
        se3_data = np.concatenate([translation_vector.cpu().detach().numpy(), quaternion])
        se3_tensor = torch.tensor(se3_data, dtype=torch.float32)
        se3_tensor = pp.SE3(se3_tensor)
        nodes_list.append(se3_tensor)
    for i in range(len(nodes_list)):
        if nodes_list[i].device != device:
            nodes_list[i] = nodes_list[i].to(device)

    nodes_list = torch.stack(nodes_list).to(torch.float32).to(device)



    graph = PoseGraph(nodes_list,config).to(device)
    solver = ppos.Cholesky()
    strategy = ppost.TrustRegion(radius=500)
    optimizer = pp.optim.LM(graph, solver=solver, strategy=strategy, min=1e-6, vectorize=True)
    scheduler = StopOnPlateau(optimizer, steps=10, patience=5, decreasing=1e-4, verbose=True)

    pngname = os.path.join(save_path, "pypose" + '.png')
    # axlim = plot_and_save(graph.nodes.translation(), pngname, name)


    while scheduler.continual():
        loss = optimizer.step(input=(edges_list,poses_list), weight=None)
        scheduler.step(loss)
        lossname = os.path.join(save_path, "pypose" + str(scheduler.steps))
        title = 'PyPose PGO at the %d step(s) with loss %7f' % (scheduler.steps, loss.item())
        # plot_and_save(graph.nodes.translation(), lossname + '.png', title, axlim=axlim)
        final_optimized_poses = graph.nodes.translation()
        torch.save(graph.state_dict(), lossname + '.pt')

    initial_poses = graph.initial_poses.translation()
    initial_poses_distance_sum = calculate_distance_sum(initial_poses, torch.stack(gt_node))
    print("Initial Poses Distance Sum:", initial_poses_distance_sum)
    final_optimized_poses_distance_sum = calculate_distance_sum(final_optimized_poses, torch.stack(gt_node))
    print("Final Optimized Poses Distance Sum:", final_optimized_poses_distance_sum)

    # 找到发生变化的帧的索引
    changed_frames = []
    for i in range(len(initial_poses)):
        if not torch.allclose(initial_poses[i], final_optimized_poses[i], atol=1e-6):
            changed_frames.append(i)

    print("Frames with changed poses:", changed_frames)

    return graph.nodes
