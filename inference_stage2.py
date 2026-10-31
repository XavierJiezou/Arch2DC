"""Arch2DC Stage II 单样本推理：预测牙冠点云 -> 牙冠网格（SAP 可微泊松重建）。

坐标归一化使用 缺损牙弓 X1 ∪ Stage I 预测牙冠 的轴对齐包围盒（与训练一致）。
可选地把重建牙冠与缺损牙弓拼接，输出完整牙弓网格。

示例：
    python inference_stage2.py --ckpt checkpoints/arch2dc/stage2_model_best.pt \
        --pred outputs/crown_pred.ply --arch path/to/defective_arch.ply \
        --output outputs/crown_mesh.ply --combined outputs/arch_restored.ply
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
SAP_DIR = os.path.join(ROOT, "shape_as_points")


def main():
    parser = argparse.ArgumentParser("Arch2DC Stage II inference")
    parser.add_argument("--ckpt", required=True, help="Stage II 权重 (model_best.pt)")
    parser.add_argument("--pred", required=True, help="Stage I 输出的预测牙冠点云 (.ply)")
    parser.add_argument("--arch", required=True, help="缺损牙弓 X1 (.ply)，用于归一化与拼接")
    parser.add_argument("--output", required=True, help="输出牙冠网格 (.ply)")
    parser.add_argument("--combined", default=None, help="可选：输出 缺损牙弓 + 重建牙冠 的拼接网格")
    parser.add_argument("--config", default="configs/learning_based/tooth_dual/arch2dc.yaml",
                        help="Stage II 配置（相对 shape_as_points/ 或绝对路径）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    # 先把用户给的路径转成绝对路径，再切换到 SAP 目录（其配置使用相对 inherit_from）
    for key in ("ckpt", "pred", "arch", "output", "combined"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, os.path.abspath(value))
    if not os.path.isabs(args.config) and os.path.exists(args.config):
        args.config = os.path.abspath(args.config)
    os.chdir(SAP_DIR)
    sys.path.insert(0, SAP_DIR)

    from src.data.tooth_dual import _load_vertices_and_normals, _sample_fixed, sap_normalizer
    from src.dpsr import DPSR
    from src.model import Encode2Points
    from src.utils import load_config, load_model_manual, mc_from_psr
    from eval_tooth_dual import sap_to_world, to_sap, transfer_vertex_colors, write_combined, write_mesh
    import open3d as o3d

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = load_config(args.config, "configs/default.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    model = Encode2Points(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device)
    load_model_manual(state["state_dict"], model)
    model.eval()
    resolution = cfg["generation"].get("psr_resolution", 0) or cfg["model"]["grid_res"]
    sigma = cfg["generation"].get("psr_sigma", 0) or cfg["model"]["psr_sigma"]
    dpsr = DPSR(res=(resolution, resolution, resolution), sig=sigma).to(device)

    pred_points, pred_normals = _load_vertices_and_normals(args.pred)
    center, scale = sap_normalizer(args.arch, pred_points)

    inputs = np.clip(to_sap(pred_points, center, scale), 0.0, 0.99).astype(np.float32)
    inputs, _ = _sample_fixed(inputs, pred_normals, cfg["data"]["pointcloud_n"])

    with torch.no_grad():
        points, normals = model(torch.from_numpy(inputs[None]).float().to(device))
        psr_grid = dpsr(points, normals)
        verts_sap, faces, _ = mc_from_psr(psr_grid, zero_level=cfg["data"].get("zero_level", 0))
    verts_world = sap_to_world(verts_sap, center, scale)

    # 牙冠颜色取自 Stage I 预测点云自带的 RGB
    pcd = o3d.io.read_point_cloud(args.pred)
    colors = None
    if pcd.has_colors():
        colors = transfer_vertex_colors(verts_world, np.asarray(pcd.points), np.asarray(pcd.colors))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    write_mesh(args.output, verts_world, faces, colors)
    print(f"Saved crown mesh ({len(verts_world)} vertices, {len(faces)} faces) -> {args.output}")
    if args.combined:
        os.makedirs(os.path.dirname(args.combined), exist_ok=True)
        write_combined(args.arch, verts_world, faces, args.combined, tooth_colors=colors)
        print(f"Saved restored arch -> {args.combined}")


if __name__ == "__main__":
    main()
