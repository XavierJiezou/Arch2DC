"""Arch2DC Stage I 单样本推理：缺损牙弓 X1 (.ply) -> 缺失牙冠点云 + RGB (.ply)。

只使用输入的缺损牙弓 X1 做归一化（质心 + 最大半径），推理过程不需要任何真值。
输出点云在输入的世界坐标系下，可直接作为 inference_stage2.py 的 --pred 输入。

示例：
    python inference_stage1.py --ckpt checkpoints/arch2dc/stage1_best_l1_cd.pth \
        --input path/to/defective_arch.ply --output outputs/crown_pred.ply
"""
import argparse
import os

import numpy as np
import open3d as o3d
import torch

from export_stage1 import build_model, save_pcd_rgb


def main():
    parser = argparse.ArgumentParser("Arch2DC Stage I inference")
    parser.add_argument("--ckpt", required=True, help="Stage I 权重 (best_l1_cd.pth)")
    parser.add_argument("--input", required=True, help="缺损牙弓 X1 (.ply 点云或网格)")
    parser.add_argument("--output", required=True, help="输出预测牙冠点云 (.ply)")
    parser.add_argument("--input_num_points", type=int, default=4096, help="输入采样点数 N")
    parser.add_argument("--gt_tooth_num_points", type=int, default=2048, help="输出点数 M")
    parser.add_argument("--num_query", type=int, default=512)
    parser.add_argument("--global_feature_dim", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0, help="GPU 编号，-1 为 CPU（模型依赖 CUDA 算子，需 GPU）")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    rng = np.random.default_rng(args.seed)

    pcd = o3d.io.read_point_cloud(args.input)
    points = np.asarray(pcd.points, dtype=np.float64)
    if len(points) == 0:
        raise ValueError(f"Empty input: {args.input}")

    # 与训练一致：用缺损牙弓 X1 自身的质心和最大半径归一化
    center = points.mean(axis=0)
    scale = float(np.linalg.norm(points - center, axis=1).max())
    idx = rng.choice(len(points), args.input_num_points, replace=len(points) < args.input_num_points)
    missing = ((points[idx] - center) / scale).astype(np.float32)

    model = build_model(args.ckpt, device, args.num_query, args.global_feature_dim,
                        args.input_num_points, args.gt_tooth_num_points)
    with torch.no_grad():
        ret = model(torch.from_numpy(missing)[None].to(device))
        pred = model.get_pred_fine(ret)[0].cpu().numpy()
        color = model.get_pred_color(ret)
        color = color[0].cpu().numpy() if color is not None else None

    pred_world = pred * scale + center
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_pcd_rgb(args.output, pred_world.astype(np.float32), color)
    print(f"Saved {len(pred_world)} predicted crown points -> {args.output}")


if __name__ == "__main__":
    main()
