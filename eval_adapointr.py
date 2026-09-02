
import os
import sys
import argparse
import numpy as np
import open3d as o3d
import torch
from models.adapointr import AdaPoinTr
from dataset import Tooth
from visualization import plot_pcd_one_view
from metrics.metric import l1_cd, l2_cd, f_score, hausdorff_distance
from torch.utils.data import DataLoader
from easydict import EasyDict
from lightning import seed_everything
import re

class DualLogger(object):
    def __init__(self, log_path):
        self.terminal = sys.__stdout__
        self.log = open(log_path, "w")

    def write(self, message):
        self.terminal.write(message)
        clean = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', message)
        self.log.write(clean)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

CATEGORIES_PCN = [str(i) for i in range(1, 10)] # 每个实验测试9次
CATEGORIES_PCN_NOVEL = ['bus', 'bed', 'bookshelf', 'bench', 'guitar', 'motorbike', 'skateboard', 'pistol']

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def export_ply(filename, points):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pc, write_ascii=True)

def test_single_category(category, model, params, save=True):
    if save:
        cat_dir = os.path.join(params.result_dir, params.exp_name, category)
        image_dir = os.path.join(cat_dir, 'image')
        output_dir = os.path.join(cat_dir, 'output')
        make_dir(cat_dir)
        make_dir(image_dir)
        make_dir(output_dir)

    test_dataset = Tooth("data/tooth", "val", half=False)
    test_dataloader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    index = 201
    total_l1_cd, total_l2_cd, total_f_score, total_hd = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for p, c in test_dataloader:
            p = p.cuda()
            c = c.cuda()
            _, c_ = model(p)
            tmp_l1_cd = l1_cd(c_, c).item()
            total_l1_cd += tmp_l1_cd
            tmp_l2_cd = l2_cd(c_, c).item()
            total_l2_cd += tmp_l2_cd
            for i in range(len(c)):
                input_pc = p[i].detach().cpu().numpy()
                output_pc = c_[i].detach().cpu().numpy()
                gt_pc = c[i].detach().cpu().numpy()
                tmp_f_score = f_score(output_pc, gt_pc)
                total_f_score += tmp_f_score
                tmp_hd = hausdorff_distance(output_pc, gt_pc)
                total_hd += tmp_hd
                print(f"Loop {category}, ID {index}, L1_CD(1e-3): {tmp_l1_cd*1e3:.4f}, L2_CD(1e-4): {tmp_l2_cd*1e4:.4f}, FScore-0.01(%): {tmp_f_score*1e2:.4f}, HD: {tmp_hd:.4f}")
                if save:
                    plot_pcd_one_view(os.path.join(image_dir, '{:03d}.jpg'.format(index)),
                                      [input_pc, output_pc, gt_pc],
                                      ['Input', 'Output', 'GT'],
                                      xlim=(-0.35, 0.35), ylim=(-0.35, 0.35), zlim=(-0.35, 0.35))
                    export_ply(os.path.join(output_dir, '{:03d}_output.ply'.format(index)), output_pc)
                    export_ply(os.path.join(output_dir, '{:03d}_input.ply'.format(index)), input_pc)
                    export_ply(os.path.join(output_dir, '{:03d}_gt.ply'.format(index)), gt_pc)
                index += 1

    avg_l1_cd = total_l1_cd / len(test_dataset)
    avg_l2_cd = total_l2_cd / len(test_dataset)
    avg_f_score = total_f_score / len(test_dataset)
    avg_hd = total_hd / len(test_dataset)

    return avg_l1_cd, avg_l2_cd, avg_f_score, avg_hd

def test(params, save=False):
    seed_everything(42, workers=True)
    if save:
        make_dir(os.path.join(params.result_dir, params.exp_name))

    print(params.exp_name)

    model = AdaPoinTr(
        EasyDict({
            "num_query": params.num_query,
            "num_points": 16384,
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
                "self_attn_block_style_list": ["attn-graph", "attn", "attn", "attn", "attn", "attn", "attn", "attn"],
                "self_attn_combine_style": "concat",
                "cross_attn_block_style_list": ["attn-graph", "attn", "attn", "attn", "attn", "attn", "attn", "attn"],
                "cross_attn_combine_style": "concat",
            }
        })
    ).cuda()
    checkpoint = params.ckpt_path if params.ckpt_path else os.path.join("logs/adapointr", params.exp_name, "checkpoints", "last_l1_cd.pth")
    print(f"Loading weights from {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model.eval()

    print('\033[33m{:20s}{:20s}{:20s}{:20s}{:20s}\033[0m'.format('Category', 'L1_CD(1e-3)', 'L2_CD(1e-4)', 'FScore-0.01(%)', 'HD'))
    print('\033[33m{:20s}{:20s}{:20s}{:20s}{:20s}\033[0m'.format('--------', '-----------', '-----------', '--------------', '-----------'))

    if params.category == 'all':
        categories = CATEGORIES_PCN_NOVEL if params.novel else CATEGORIES_PCN
        l1_cds, l2_cds, fscores, hd_list = [], [], [], []
        for category in categories:
            avg_l1_cd, avg_l2_cd, avg_f_score, avg_hd = test_single_category(category, model, params, save)
            print('{:20s}{:<20.4f}{:<20.4f}{:<20.4f}{:<20.4f}'.format(
                category.title(), 1e3 * avg_l1_cd, 1e4 * avg_l2_cd, 1e2 * avg_f_score, avg_hd))
            l1_cds.append(avg_l1_cd)
            l2_cds.append(avg_l2_cd)
            fscores.append(avg_f_score)
            hd_list.append(avg_hd)
        print('\033[33m{:20s}{:20s}{:20s}{:20s}{:20s}\033[0m'.format('--------', '-----------', '-----------', '--------------', '-----------'))
        print('\033[32m{:20s}{:<20.4f}{:<20.4f}{:<20.4f}{:<20.4f}\033[0m'.format(
            'Average', np.mean(l1_cds) * 1e3, np.mean(l2_cds) * 1e4, np.mean(fscores) * 1e2, np.mean(hd_list)))
    else:
        avg_l1_cd, avg_l2_cd, avg_f_score, avg_hd = test_single_category(params.category, model, params, save)
        print('{:20s}{:<20.4f}{:<20.4f}{:<20.4f}{:<20.4f}'.format(
            params.category.title(), 1e3 * avg_l1_cd, 1e4 * avg_l2_cd, 1e2 * avg_f_score, avg_hd))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Point Cloud Completion Testing')
    parser.add_argument('--exp_name', type=str, help='Tag of experiment')
    parser.add_argument('--result_dir', type=str, default='results/adapointr', help='Results directory')
    parser.add_argument('--ckpt_path', type=str, default=None, help='The path of pretrained model.')
    parser.add_argument('--category', type=str, default='all', help='Category of point clouds')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loader')
    parser.add_argument('--num_workers', type=int, default=4, help='Num workers for data loader')
    parser.add_argument('--save', type=bool, default=False, help='Saving test result')
    parser.add_argument('--novel', type=bool, default=False, help='unseen categories for testing')
    parser.add_argument('--emd', type=bool, default=False, help='Whether evaluate emd')
    parser.add_argument('--num_query', type=int, default=4, help='Model grid size')
    parser.add_argument('--global_feature_dim', type=int, default=1024, help='Model latent dimension')
    params = parser.parse_args()

    log_dir = os.path.join(params.result_dir, params.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "result.log")
    sys.stdout = DualLogger(log_file)

    test(params, params.save)
