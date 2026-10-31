"""Arch2DC Stage I 评估：默认在 test 划分上评估；模型输入按 X1 归一化，指标换算到完整牙弓 X0 的质心/最大半径坐标系下计算。"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from easydict import EasyDict
from lightning import seed_everything

from dataset import ToothNoMask
from models import ToothNoMaskAdaPoinTr
from metrics.metric import l1_cd, l2_cd, f_score, hausdorff_distance
from visualization import plot_pcd_one_view


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def symmetric_color_nn_l1(pred_xyz, pred_rgb, gt_xyz, gt_rgb):
    d = torch.cdist(pred_xyz, gt_xyz)
    idx_pg = torch.argmin(d, dim=2)
    idx_gp = torch.argmin(d, dim=1)

    gt_rgb_nn = torch.gather(gt_rgb, 1, idx_pg.unsqueeze(-1).expand(-1, -1, 3))
    pred_rgb_nn = torch.gather(pred_rgb, 1, idx_gp.unsqueeze(-1).expand(-1, -1, 3))

    loss_pg = torch.abs(pred_rgb - gt_rgb_nn).mean()
    loss_gp = torch.abs(gt_rgb - pred_rgb_nn).mean()
    return 0.5 * (loss_pg + loss_gp)


def export_ply(filename, points, colors=None):
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {points.shape[0]}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        if colors is not None:
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
        f.write('end_header\n')
        if colors is None:
            for p in points:
                f.write(f'{p[0]} {p[1]} {p[2]}\n')
        else:
            colors_255 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
            for p, c in zip(points, colors_255):
                f.write(f'{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n')


def evaluate(params, save=False):
    seed_everything(42, workers=True)

    if torch.cuda.is_available() and params.gpu >= 0:
        torch.cuda.set_device(params.gpu)
        device = torch.device(f"cuda:{params.gpu}")
    else:
        device = torch.device("cpu")

    if save:
        make_dir(os.path.join(params.result_dir, params.exp_name))

    test_dataset = ToothNoMask(
        dataroot=params.data_root,
        split=params.split,
        missing_num_points=params.missing_num_points,
        gt_tooth_num_points=params.gt_tooth_num_points,
        complete_num_points=params.complete_num_points,
        train_ratio=params.train_ratio,
        val_ratio=params.val_ratio,
        test_ratio=params.test_ratio,
        return_colors=params.use_color,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=True,
    )

    model = ToothNoMaskAdaPoinTr(
        EasyDict({
            "num_query": params.num_query,
            "num_points": params.gt_tooth_num_points,
            "center_num": [512, 256],
            "global_feature_dim": params.global_feature_dim,
            "encoder_type": "graph",
            "decoder_type": "fc",
            "encoder_config": {
                "embed_dim": 384,
                "depth": 6,
                "num_heads": 6,
                "k": 8,
                "n_group": 2,
                "mlp_ratio": 2.0,
                "block_style_list": ["attn-graph", "attn", "attn", "attn", "attn", "attn"],
                "combine_style": "concat",
            },
            "decoder_config": {
                "embed_dim": 384,
                "depth": 8,
                "num_heads": 6,
                "k": 8,
                "n_group": 2,
                "mlp_ratio": 2.0,
                "self_attn_block_style_list": [
                    "attn-graph",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                ],
                "self_attn_combine_style": "concat",
                "cross_attn_block_style_list": [
                    "attn-graph",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                ],
                "cross_attn_combine_style": "concat",
            },
        }),
        input_num_points=params.input_num_points,
        enable_color=params.use_color,
    ).to(device)

    checkpoint = params.ckpt_path if params.ckpt_path else os.path.join(
        "logs/adapointr", params.exp_name, "checkpoints", "best_l1_cd.pth"
    )
    print(f"Loading weights from {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=not params.use_color)
    model.eval()

    total_l1_tooth, total_l2_tooth = 0.0, 0.0
    total_l1_full, total_l2_full = 0.0, 0.0
    total_total = 0.0
    total_fscore, total_hd = 0.0, 0.0
    total_fscore_full, total_hd_full = 0.0, 0.0
    total_color_l1 = 0.0

    if save:
        cat_dir = os.path.join(params.result_dir, params.exp_name, 'tooth_nomask')
        image_dir = os.path.join(cat_dir, 'image')
        output_dir = os.path.join(cat_dir, 'output')
        make_dir(cat_dir)
        make_dir(image_dir)
        make_dir(output_dir)

    index = 0
    with torch.no_grad():
        for batch in test_dataloader:
            missing_pc = batch['missing_pc'].to(device)
            gt_tooth_pc = batch['gt_tooth_pc'].to(device)
            complete_pc = batch['complete_pc'].to(device)
            gt_tooth_color = batch['gt_tooth_color'].to(device) if params.use_color else None

            ret = model(missing_pc)
            pred_fine = model.get_pred_fine(ret)
            pred_color = model.get_pred_color(ret)

            # 指标坐标系：把 X1 归一化坐标换算到 X0 的质心/最大半径坐标系（仅用于算指标）
            c = batch['norm_center'].to(device)[:, None, :]
            s = batch['norm_scale'].to(device).view(-1, 1, 1)
            ec = batch['eval_center'].to(device)[:, None, :]
            es = batch['eval_scale'].to(device).view(-1, 1, 1)
            to_metric = lambda p: (p * s + c - ec) / es
            missing_pc, gt_tooth_pc, complete_pc, pred_fine = (
                to_metric(missing_pc), to_metric(gt_tooth_pc), to_metric(complete_pc), to_metric(pred_fine))

            l1_tooth = l1_cd(pred_fine, gt_tooth_pc)
            l2_tooth = l2_cd(pred_fine, gt_tooth_pc)

            merged_pc = torch.cat([missing_pc, pred_fine], dim=1)
            l1_full = l1_cd(merged_pc, complete_pc)
            l2_full = l2_cd(merged_pc, complete_pc)

            total_l1_tooth += l1_tooth.item()
            total_l2_tooth += l2_tooth.item()
            total_l1_full += l1_full.item()
            total_l2_full += l2_full.item()
            total_total += (params.local_weight * l1_tooth + params.full_weight * l1_full).item()
            if params.use_color:
                total_color_l1 += symmetric_color_nn_l1(pred_fine, pred_color, gt_tooth_pc, gt_tooth_color).item()

            for i in range(pred_fine.shape[0]):
                pred_np = pred_fine[i].detach().cpu().numpy()
                gt_np = gt_tooth_pc[i].detach().cpu().numpy()
                total_fscore += f_score(pred_np, gt_np)
                total_hd += hausdorff_distance(pred_np, gt_np)
                merged_np = merged_pc[i].detach().cpu().numpy()
                complete_np = complete_pc[i].detach().cpu().numpy()
                total_fscore_full += f_score(merged_np, complete_np)
                total_hd_full += hausdorff_distance(merged_np, complete_np)

                if save:
                    completed_np = torch.cat([missing_pc[i], pred_fine[i]], dim=0).detach().cpu().numpy()
                    plot_pcd_one_view(
                        os.path.join(image_dir, f'{index:03d}.jpg'),
                        [
                            missing_pc[i].detach().cpu().numpy(),
                            completed_np,
                            pred_np,
                            gt_np,
                            complete_pc[i].detach().cpu().numpy(),
                        ],
                        ['Missing Jaw', 'Completed Jaw', 'Pred Tooth', 'GT Tooth', 'GT Jaw'],
                        xlim=(-0.35, 0.35), ylim=(-0.35, 0.35), zlim=(-0.35, 0.35)
                    )
                    if params.use_color:
                        pred_rgb_np = pred_color[i].detach().cpu().numpy()
                        gt_rgb_np = gt_tooth_color[i].detach().cpu().numpy()
                        export_ply(os.path.join(output_dir, f'{index:03d}_pred_tooth.ply'), pred_np, pred_rgb_np)
                        export_ply(os.path.join(output_dir, f'{index:03d}_gt_tooth.ply'), gt_np, gt_rgb_np)
                    else:
                        export_ply(os.path.join(output_dir, f'{index:03d}_pred_tooth.ply'), pred_np)
                        export_ply(os.path.join(output_dir, f'{index:03d}_gt_tooth.ply'), gt_np)
                index += 1

    num_samples = len(test_dataset)
    if num_samples > 0:
        avg_l1_tooth = total_l1_tooth / num_samples
        avg_l2_tooth = total_l2_tooth / num_samples
        avg_l1_full = total_l1_full / num_samples
        avg_l2_full = total_l2_full / num_samples
        avg_total = total_total / num_samples
        avg_fscore = total_fscore / num_samples
        avg_hd = total_hd / num_samples
        avg_fscore_full = total_fscore_full / num_samples
        avg_hd_full = total_hd_full / num_samples
        avg_color_l1 = total_color_l1 / num_samples
    else:
        avg_l1_tooth = avg_l2_tooth = 0.0
        avg_l1_full = avg_l2_full = 0.0
        avg_total = 0.0
        avg_fscore = avg_hd = 0.0
        avg_fscore_full = avg_hd_full = 0.0
        avg_color_l1 = 0.0

    print('ToothNoMask Evaluation')
    print('{:20s}{:20s}{:20s}{:20s}{:20s}{:20s}{:20s}{:20s}{:20s}'.format(
        'Metric', 'L1_Tooth(1e-3)', 'L2_Tooth(1e-4)', 'FScore_S(%)', 'HD_S',
        'L1_Full(1e-3)', 'L2_Full(1e-4)', 'FScore_F(%)', 'HD_F'
    ))
    print('{:20s}{:<20.4f}{:<20.4f}{:<20.4f}{:<20.4f}{:<20.4f}{:<20.4f}{:<20.4f}{:<20.4f}'.format(
        'Average', avg_l1_tooth * 1e3, avg_l2_tooth * 1e4, avg_fscore * 1e2, avg_hd,
        avg_l1_full * 1e3, avg_l2_full * 1e4, avg_fscore_full * 1e2, avg_hd_full
    ))
    print('{:20s}{:<20.4f}'.format('Total Loss(1e-3)', avg_total * 1e3))
    if params.use_color:
        print('{:20s}{:<20.4f}'.format('Color L1(1e-3)', avg_color_l1 * 1e3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ToothNoMaskAdaPoinTr Evaluation')
    parser.add_argument('--exp_name', type=str, default='arch2dc_stage1', help='Tag of experiment')
    parser.add_argument('--result_dir', type=str, default='results/adapointr', help='Results directory')
    parser.add_argument('--ckpt_path', type=str, default=None, help='The path of pretrained model')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loader')
    parser.add_argument('--num_workers', type=int, default=4, help='Num workers for data loader')
    parser.add_argument('--save', action='store_true', help='Saving test result')

    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='Dataset split to evaluate')

    parser.add_argument('--num_query', type=int, default=512, help='num_query')
    parser.add_argument('--global_feature_dim', type=int, default=1024, help='Model latent dimension')

    parser.add_argument('--data_root', type=str, default='data/tooth2', help='Dataset root')
    parser.add_argument('--missing_num_points', type=int, default=4096, help='Missing upper jaw points')
    parser.add_argument('--gt_tooth_num_points', type=int, default=2048, help='GT tooth points')
    parser.add_argument('--complete_num_points', type=int, default=8192, help='Complete jaw points')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Train split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Val split ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test split ratio')
    parser.add_argument('--local_weight', type=float, default=0.8, help='Weight for local single-tooth loss')
    parser.add_argument('--full_weight', type=float, default=0.2, help='Weight for full jaw loss')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index, set -1 for CPU')
    parser.add_argument('--input_num_points', type=int, default=4096, help='Total input points')
    parser.add_argument('--use_color', action='store_true', help='Enable RGB completion branch')

    params = parser.parse_args()
    evaluate(params, params.save)
