"""
用训练好的 Stage I 模型为 train/val/test 导出预测牙冠点云（带 RGB），作为 Stage II 的输入。
输出：{out_dir}/{split}/predtooth/{id}_predtooth.ply（世界坐标）
列表：{out_dir}/lists/{split}.lst，每行：predtooth.ply  完整牙弓_0.ply  GT牙冠_3.ply（绝对路径）
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from easydict import EasyDict
from lightning import seed_everything
from torch.utils.data import DataLoader

from dataset.tooth_nomask import ToothNoMask
from models import ToothNoMaskAdaPoinTr


def save_pcd_rgb(path, points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0).astype(np.float64))
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)


def build_model(ckpt_path, device, num_query=512, global_feature_dim=1024, input_num_points=4096, gt_tooth_num_points=2048):
    model = ToothNoMaskAdaPoinTr(
        EasyDict({
            "num_query": num_query,
            "num_points": gt_tooth_num_points,
            "center_num": [512, 256],
            "global_feature_dim": global_feature_dim,
            "encoder_type": "graph",
            "decoder_type": "fc",
            "encoder_config": {
                "embed_dim": 384, "depth": 6, "num_heads": 6, "k": 8, "n_group": 2,
                "mlp_ratio": 2.0,
                "block_style_list": ["attn-graph", "attn", "attn", "attn", "attn", "attn"],
                "combine_style": "concat",
            },
            "decoder_config": {
                "embed_dim": 384, "depth": 8, "num_heads": 6, "k": 8, "n_group": 2,
                "mlp_ratio": 2.0,
                "self_attn_block_style_list": ["attn-graph"] + ["attn"] * 7,
                "self_attn_combine_style": "concat",
                "cross_attn_block_style_list": ["attn-graph"] + ["attn"] * 7,
                "cross_attn_combine_style": "concat",
            },
        }),
        input_num_points=input_num_points,
        enable_color=True,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Loaded checkpoint: {ckpt_path}")
    return model


def infer_split(model, split, args, device, out_root, data_root):
    dataset = ToothNoMask(
        dataroot=args.data_dir,
        split=split,
        missing_num_points=args.input_num_points,
        gt_tooth_num_points=args.gt_tooth_num_points,
        complete_num_points=args.complete_num_points,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        return_colors=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    pred_dir = out_root / split / "predtooth"
    pred_dir.mkdir(parents=True, exist_ok=True)

    triplets = []
    for i, batch in enumerate(loader):
        missing = batch["missing_pc"].to(device)
        norm_center = batch["norm_center"][0].numpy()   # (3,)
        norm_scale  = float(batch["norm_scale"][0].item())

        with torch.no_grad():
            ret = model(missing)
            pred_tooth = model.get_pred_fine(ret)[0].cpu().numpy()   # (N, 3)
            pred_color = model.get_pred_color(ret)
            if pred_color is not None:
                pred_color = pred_color[0].cpu().numpy()              # (N, 3)

        # 反归一化到原始坐标系
        pred_raw = pred_tooth * norm_scale + norm_center

        stem = Path(dataset.files[i]).stem   # e.g. "100_0"
        base_id = stem.rsplit("_", 1)[0]     # "100"
        pred_path = pred_dir / f"{base_id}_predtooth.ply"
        save_pcd_rgb(pred_path, pred_raw.astype(np.float32), pred_color)

        file0 = data_root / f"{base_id}_0.ply"
        file3 = data_root / f"{base_id}_3.ply"
        if file0.exists() and file3.exists():
            triplets.append(f"{pred_path.resolve()} {file0.resolve()} {file3.resolve()}")
        else:
            print(f"  WARNING: {base_id} missing file0 or file3, skipped")

        if (i + 1) % 20 == 0 or (i + 1) == len(dataset):
            print(f"  [{split}] {i+1}/{len(dataset)}")

    lst_dir = out_root / "lists"
    lst_dir.mkdir(parents=True, exist_ok=True)
    (lst_dir / f"{split}.lst").write_text("\n".join(triplets) + "\n")
    print(f"  [{split}] Saved {len(triplets)} entries → {lst_dir}/{split}.lst")


def main():
    parser = argparse.ArgumentParser("生成带 RGB 的 predtooth PLY 供 SAP 使用")
    parser.add_argument("--ckpt", required=True, help="AdaPoinTr checkpoint 路径")
    parser.add_argument("--out_dir", default="results/stage1_pred", help="输出目录")
    parser.add_argument("--data_dir", default="data/tooth2")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--num_query", type=int, default=512)
    parser.add_argument("--global_feature_dim", type=int, default=1024)
    parser.add_argument("--input_num_points", type=int, default=4096)
    parser.add_argument("--gt_tooth_num_points", type=int, default=2048)
    parser.add_argument("--complete_num_points", type=int, default=8192)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    seed_everything(42, workers=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    out_root = Path(args.out_dir)
    data_root = Path(args.data_dir)
    model = build_model(args.ckpt, device, args.num_query, args.global_feature_dim,
                        args.input_num_points, args.gt_tooth_num_points)

    for split in args.splits:
        print(f"\n=== {split} ===")
        infer_split(model, split, args, device, out_root, data_root)

    print(f"\n全部完成。输出目录：{out_root}")


if __name__ == "__main__":
    main()
