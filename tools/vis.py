import numpy as np
import os
import imageio
import open3d as o3d
from tqdm import tqdm
import sys
import torch
import trimesh

from utils import extract_mesh, getVoxels, get_batch_query_fn


def load_poses_from_folder(folder_path):
    """
    Load all 4x4 pose matrices from .txt files in the specified folder.
    """
    poses = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            pose_matrix = np.loadtxt(file_path)  # Load the 4x4 matrix from the txt file
            if pose_matrix.shape == (4, 4):
                poses.append(pose_matrix)
            else:
                print(f"Error: Pose in {file_path} is not a valid 4x4 matrix")
    return poses


def apply_colors_to_point_cloud(pcd, color_func=None, config=None, bounding_box=None):
    """
    Apply color to the point cloud. If a color function is provided, use it to generate the colors.

    Parameters:
    - pcd: The open3d point cloud object.
    - color_func: Optional function to generate colors for the point cloud.

    Returns:
    - The point cloud with colors applied.
    """
    points = np.asarray(pcd.points)

    if color_func is not None:
        if not config['mesh']['render_color']:
            if config['grid']['tcnn_encoding']:
                # Normalize points according to the bounding box
                vert_flat = (torch.from_numpy(points).to(torch.device("cuda")) - bounding_box[:, 0]) / (
                            bounding_box[:, 1] - bounding_box[:, 0])

            # Get the batch processing function for color
            fn_color = get_batch_query_fn(color_func, 1)

            chunk_size = 1024 * 64
            raw_colors = [fn_color(vert_flat, i, min(i + chunk_size, vert_flat.shape[0])).cpu().numpy() for i in
                          range(0, vert_flat.shape[0], chunk_size)]
            colors = np.concatenate(raw_colors, 0)

            # Reshape the colors to match the point cloud format
            colors = np.reshape(colors, (-1, 3))
        elif config['mesh']['render_color']:
            print('rendering surface color')
            vertex_normals = torch.from_numpy(np.asarray(pcd.normals))  # assuming the point cloud has normals
            fn_color = get_batch_query_fn(color_func, 2)

            chunk_size = 1024 * 64
            raw_colors = [fn_color(torch.from_numpy(points), vertex_normals, i,
                                   min(i + chunk_size, points.shape[0])).cpu().numpy() for i in
                          range(0, points.shape[0], chunk_size)]
            colors = np.concatenate(raw_colors, 0)

            # Reshape the colors to match the point cloud format
            colors = np.reshape(colors, (-1, 3))
        else:
            # Default to random colors if no color function is provided
            colors = np.random.rand(points.shape[0], 3)
    else:
        # Default to random colors if no color function is provided
        colors = np.random.rand(points.shape[0], 3)

    pcd.colors = o3d.utility.Vector3dVector(colors)  # Assign colors to the point cloud
    return pcd


def render_point_clouds_and_poses(point_clouds, poses_folder, output_folder, color_func=None):
    """
    Render all point clouds and camera poses in a single visualization.

    Parameters:
    - point_clouds: List of numpy arrays, where each array represents a point cloud.
    - poses_folder: Folder path that contains .txt files with 4x4 pose matrices.
    - output_folder: Folder path to save the rendered image.
    - color_func: Optional function to apply colors to point clouds.
    """
    # Load poses from the specified folder
    poses = load_poses_from_folder(poses_folder)

    # Create a new visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Clouds and Poses", width=1920, height=1080)

    # Add all point clouds to the visualizer with colors
    for i, pc in enumerate(point_clouds):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)

        # Apply colors to the point cloud
        pcd = apply_colors_to_point_cloud(pcd, color_func)

        vis.add_geometry(pcd)

    # Add all camera poses as line sets
    for i, pose in enumerate(poses):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Simple coordinate system
        lines = np.array([[0, 1], [0, 2], [0, 3]])  # Lines for x, y, z axes
        lc = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines)
        )
        lc.transform(pose)  # Apply the pose transformation
        lc.paint_uniform_color([0.0, 1.0, 0.0])  # Green color for lines
        vis.add_geometry(lc)

    # Run the visualizer
    vis.run()
    vis.destroy_window()

    # Save the rendered image
    save_path = f"{output_folder}/combined_rendering.png"
    vis.capture_screen_image(save_path)
    print(f"Rendered image saved to {save_path}")
if __name__ == "__main__":
    # Command line arguments: point_cloud_file, poses_folder, output_folder
    point_cloud_file = sys.argv[1]  # First argument: .ply file
    poses_folder = sys.argv[2]      # Second argument: poses folder path
    output_folder = sys.argv[3]     # Third argument: output folder path

    # Load the point cloud from the .ply file
    pcd = o3d.io.read_point_cloud(point_cloud_file)
    point_clouds = [np.asarray(pcd.points)]

    # Call the render function
    render_point_clouds_and_poses(point_clouds, poses_folder, output_folder)