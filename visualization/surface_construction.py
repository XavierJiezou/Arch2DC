import os
import sys
import shutil
import tempfile
import subprocess
import open3d as o3d
import numpy as np


def convert_ply_to_stl(ply_path, stl_path, ckpt_epoch=300, gpu=0):
    """使用 CircNet 进行表面重建，然后保存/可视化结果网格。

    流程：
    1. 将输入点云拷贝到临时目录，作为 CircNet main_evaluate.py 的输入。
    2. 在 CircNet/codes 下调用 main_evaluate.py 生成 primitive 网格；
    3. 调用 run_prim2manifold.py 将 primitive 提升为边界流形网格；
    4. 读取对应输出网格，保存到 stl_path，并用 Open3D 可视化。

    参数:
        ply_path (str): 输入的 .ply 点云文件路径。
        stl_path (str): 输出网格文件路径（扩展名可为 .ply/.stl 等）。
        ckpt_epoch (int): 加载的 CircNet 检查点 epoch，例如 80 或 300。
        gpu (int): 使用的 GPU 索引（与 CircNet main_evaluate 一致）。
    """
    print(f"处理中（CircNet 重建）: {ply_path}")

    if not os.path.exists(ply_path):
        print(f"❌ 输入点云不存在：{ply_path}")
        return

    # 计算 CircNet 工程路径
    # 为避免 cwd 等因素造成偏差，这里直接假定 CircNet 在当前用户家目录下：~/CircNet
    # 这与你给出的实际路径 /home/xxx/CircNet 一致
    home_dir = os.path.expanduser("~")
    circnet_root = os.path.join(home_dir, "CircNet")
    circnet_codes_dir = os.path.join(circnet_root, "codes")
    circnet_log_dir = os.path.join(circnet_root, "log_abc")
    circnet_results_root = os.path.join(circnet_root, "results_from_tooth")

    if not os.path.isdir(circnet_codes_dir):
        print(f"❌ 未找到 CircNet 工程目录: {circnet_codes_dir}")
        print("请确认 CircNet 与 tooth-surface-reconstruction 位于同一级目录下。")
        return

    os.makedirs(circnet_results_root, exist_ok=True)

    # 将当前点云拷贝到一个临时目录，作为 main_evaluate.py 的 pcd_dir
    temp_pcd_dir = tempfile.mkdtemp(prefix="circnet_pcd_")
    temp_ply_path = os.path.join(temp_pcd_dir, os.path.basename(ply_path))
    shutil.copy2(ply_path, temp_ply_path)

    try:
        # 调用 CircNet 的 main_evaluate.py 生成 primitive 网格
        print("➡ 调用 CircNet main_evaluate.py 进行网格划分 ...")
        cmd_eval = [
            sys.executable,
            "main_evaluate.py",
            "--gpu", str(gpu),
            "--pcd_dir", temp_pcd_dir,
            "--log_dir", circnet_log_dir,
            "--write_dir", circnet_results_root,
            "--ckpt_epoch", str(ckpt_epoch),
        ]
        subprocess.run(cmd_eval, cwd=circnet_codes_dir, check=True)

        # 调用 run_prim2manifold.py，将 primitive -> manifold
        primitive_dir = os.path.join(circnet_results_root, "primitive")
        print("➡ 调用 CircNet run_prim2manifold.py 生成流形网格 ...")
        cmd_post = [
            sys.executable,
            "run_prim2manifold.py",
            "--read_path", primitive_dir,
        ]
        subprocess.run(cmd_post, cwd=circnet_codes_dir, check=True)

    except subprocess.CalledProcessError as e:
        print(f"❌ 调用 CircNet 脚本失败，返回码 {e.returncode}")
        return
    finally:
        # 清理临时输入点云目录
        shutil.rmtree(temp_pcd_dir, ignore_errors=True)

    # 从 CircNet 输出目录中读取对应的 manifold 网格
    manifold_dir = os.path.join(circnet_results_root, "manifold")
    output_mesh_path = os.path.join(manifold_dir, os.path.basename(ply_path))

    if not os.path.exists(output_mesh_path):
        print(f"❌ 未找到 CircNet 输出网格：{output_mesh_path}")
        return

    mesh = o3d.io.read_triangle_mesh(output_mesh_path)
    if len(mesh.triangles) == 0:
        print(f"⚠️ CircNet 输出网格为空：{output_mesh_path}")
        return

    mesh.compute_vertex_normals()

    # 保存到指定路径（可为 .ply/.stl 等）
    os.makedirs(os.path.dirname(os.path.abspath(stl_path)), exist_ok=True)
    success = o3d.io.write_triangle_mesh(stl_path, mesh)

    if success:
        print(f"✅ 重建网格已保存: {stl_path}")
    else:
        print(f"❌ 保存重建网格失败: {stl_path}")

    # 可视化最终网格（在无图形界面的服务器上可能会失败，此时仅提示路径）
    print("➡ 尝试打开可视化窗口展示 CircNet 重建结果 ...")
    try:
        o3d.visualization.draw_geometries([mesh], window_name="CircNet Reconstruction")
    except Exception as e:
        print("⚠️ Open3D 可视化失败（可能是无显示环境）：", e)
        print("   请在本地图形环境中使用 MeshLab / CloudCompare / Open3D GUI 打开: ")
        print(f"   {stl_path}")


# 示例调用
if __name__ == '__main__':
    # 直接使用给出的绝对路径作为示例输入
    example_input = "xxx.ply"

    # 输出放在当前脚本所在工程下
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(this_dir))
    example_output = os.path.join(project_root, "visualization/surface_construction/xxx_circnet.ply")

    convert_ply_to_stl(example_input, example_output, ckpt_epoch=300, gpu=1)