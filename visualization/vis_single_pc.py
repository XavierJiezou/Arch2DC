import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def o3d_visualize_pc(pc):
    """
    Visualize point cloud using Open3D.
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([point_cloud])


def plot_pcd_one_view(filename, pcds, titles, suptitle='', sizes=None, cmap='Reds', zdir='y',
                      xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5, 0.5)):
    """
    Save point cloud visualization to an image using Matplotlib.
    """
    if sizes is None:
        sizes = [0.5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3 * 1.4, 3 * 1.4))
    elev = 60  # Elevation angle
    azim = -90  # Azimuth angle
    for j, (pcd, size) in enumerate(zip(pcds, sizes)):
        color = pcd[:, 0]
        ax = fig.add_subplot(1, len(pcds), j + 1, projection='3d')
        ax.view_init(elev, azim)
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=-1.0, vmax=0.5)
        # ax.set_title(titles[j])
        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    # plt.suptitle(suptitle)
    plt.tight_layout(pad=0)
    fig.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


if __name__ == '__main__':
    # Path to the PLY file
    # ply_file = "results/adapointr/epoch400/tooth_one_adapointr_num_query_1024_global_feature_dim_1024_bs4_e400/2/output/207_gt.ply"
    ply_file = "/home/zouxuechao/tooth-surface-reconstruction/data/tooth2/297_1.ply"

    # Check if file exists
    if not os.path.exists(ply_file):
        print(f"Error: File {ply_file} does not exist.")
        exit()

    # Load point cloud using Open3D
    point_cloud = o3d.io.read_point_cloud(ply_file)

    # Check if point cloud is valid
    if point_cloud.is_empty():
        print(f"Error: Failed to load point cloud from {ply_file}.")
        exit()

    # Convert point cloud to numpy array for visualization
    points = np.asarray(point_cloud.points)

    # Visualize the point cloud using Open3D
    o3d_visualize_pc(points)

    # Save the point cloud visualization as an image
    output_image = "tmp.jpg"
    plot_pcd_one_view(
        filename=output_image,
        pcds=[points],
        titles=["Point Cloud Visualization"],
        suptitle="PLY Point Cloud",
        xlim=(-0.5, 0.5),
        ylim=(-0.5, 0.5),
        zlim=(-0.5, 0.5)
    )
    print(f"Point cloud visualization saved to {output_image}.")