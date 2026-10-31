"""Arch2DC Stage I：缺失牙冠点云 + RGB 补全（AdaPoinTr + 局部/全局双约束）。

损失 = local_weight * CD(预测牙, GT牙) + full_weight * CD(缺损牙弓 ∪ 预测牙, 完整牙弓) + color_weight * 颜色L1
论文设置 α=0.8：--local_weight 0.8 --full_weight 0.2 --use_color
"""
import argparse
import os
import datetime
import random

import torch
import torch.optim as Optim

from torch.utils.data.dataloader import DataLoader
from tensorboardX import SummaryWriter

from dataset import ToothNoMask
from models import ToothNoMaskAdaPoinTr
from metrics.metric import l1_cd
from visualization import plot_pcd_one_view
from easydict import EasyDict
from lightning import seed_everything


def symmetric_color_nn_l1(pred_xyz, pred_rgb, gt_xyz, gt_rgb):
    # pred_xyz/pred_rgb: (B, Np, 3), gt_xyz/gt_rgb: (B, Ng, 3)
    d = torch.cdist(pred_xyz, gt_xyz)

    idx_pg = torch.argmin(d, dim=2)  # (B, Np)
    idx_gp = torch.argmin(d, dim=1)  # (B, Ng)

    gt_rgb_nn = torch.gather(gt_rgb, 1, idx_pg.unsqueeze(-1).expand(-1, -1, 3))
    pred_rgb_nn = torch.gather(pred_rgb, 1, idx_gp.unsqueeze(-1).expand(-1, -1, 3))

    loss_pg = torch.abs(pred_rgb - gt_rgb_nn).mean()
    loss_gp = torch.abs(gt_rgb - pred_rgb_nn).mean()
    return 0.5 * (loss_pg + loss_gp)


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def log(fd, message, time=True):
    if time:
        message = ' ==> '.join([datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), message])
    fd.write(message + '\n')
    fd.flush()
    print(message)


def prepare_logger(params):
    make_dir(params.log_dir)
    make_dir(os.path.join(params.log_dir, params.exp_name))

    logger_path = os.path.join(params.log_dir, params.exp_name)
    ckpt_dir = os.path.join(params.log_dir, params.exp_name, 'checkpoints')
    epochs_dir = os.path.join(params.log_dir, params.exp_name, 'epochs')

    make_dir(logger_path)
    make_dir(ckpt_dir)
    make_dir(epochs_dir)

    logger_file = os.path.join(params.log_dir, params.exp_name, 'logger.log')
    log_fd = open(logger_file, 'a')

    log(log_fd, f"Experiment: {params.exp_name}", False)
    log(log_fd, f"Logger directory: {logger_path}", False)
    log(log_fd, str(params), False)

    train_writer = SummaryWriter(os.path.join(logger_path, 'train'))
    val_writer = SummaryWriter(os.path.join(logger_path, 'val'))

    return ckpt_dir, epochs_dir, log_fd, train_writer, val_writer


def train(params):
    seed_everything(params.seed, workers=True)
    torch.backends.cudnn.benchmark = True

    ckpt_dir, epochs_dir, log_fd, train_writer, val_writer = prepare_logger(params)

    log(log_fd, 'Loading Data...')

    train_dataset = ToothNoMask(
        dataroot=params.data_root,
        split='train',
        missing_num_points=params.missing_num_points,
        gt_tooth_num_points=params.gt_tooth_num_points,
        complete_num_points=params.complete_num_points,
        train_ratio=params.train_ratio,
        val_ratio=params.val_ratio,
        test_ratio=params.test_ratio,
        return_colors=params.use_color,
    )
    val_dataset = ToothNoMask(
        dataroot=params.data_root,
        split='val',
        missing_num_points=params.missing_num_points,
        gt_tooth_num_points=params.gt_tooth_num_points,
        complete_num_points=params.complete_num_points,
        train_ratio=params.train_ratio,
        val_ratio=params.val_ratio,
        test_ratio=params.test_ratio,
        return_colors=params.use_color,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers, pin_memory=True)
    log(log_fd, "Dataset loaded!")

    if torch.cuda.is_available() and params.gpu >= 0:
        torch.cuda.set_device(params.gpu)
        device = torch.device(f"cuda:{params.gpu}")
    else:
        device = torch.device("cpu")

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
                "self_attn_block_style_list": ["attn-graph", "attn", "attn", "attn", "attn", "attn", "attn", "attn"],
                "self_attn_combine_style": "concat",
                "cross_attn_block_style_list": ["attn-graph", "attn", "attn", "attn", "attn", "attn", "attn", "attn"],
                "cross_attn_combine_style": "concat",
            }
        }),
        input_num_points=params.input_num_points,
        enable_color=params.use_color,
    ).to(device)

    optimizer = Optim.Adam(model.parameters(), lr=params.lr, betas=(0.9, 0.999))
    lr_schedual = Optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)

    step = max(1, len(train_dataloader) // params.log_frequency)

    if params.ckpt_path is not None:
        ckpt = torch.load(params.ckpt_path)
        model.load_state_dict(ckpt, strict=not params.use_color)

    best_cd_l1 = 1e8
    best_epoch_l1 = -1
    train_step, val_step = 0, 0

    for epoch in range(1, params.epochs + 1):
        model.train()
        for i, batch in enumerate(train_dataloader):
            missing_pc = batch['missing_pc'].to(device)
            gt_tooth_pc = batch['gt_tooth_pc'].to(device)
            complete_pc = batch['complete_pc'].to(device)
            missing_color = batch['missing_color'].to(device) if params.use_color else None
            gt_tooth_color = batch['gt_tooth_color'].to(device) if params.use_color else None

            optimizer.zero_grad()

            ret = model(missing_pc)
            pred_fine = model.get_pred_fine(ret)
            pred_color = model.get_pred_color(ret)

            loss_tooth = l1_cd(pred_fine, gt_tooth_pc)
            merged_pc = torch.cat([missing_pc, pred_fine], dim=1)
            loss_full = l1_cd(merged_pc, complete_pc)
            loss = params.local_weight * loss_tooth + params.full_weight * loss_full

            color_loss = None
            if params.use_color:
                color_loss = symmetric_color_nn_l1(pred_fine, pred_color, gt_tooth_pc, gt_tooth_color)
                loss = loss + params.color_weight * color_loss

            loss.backward()
            optimizer.step()

            if (i + 1) % step == 0:
                if params.use_color:
                    log(log_fd, "Training Epoch [{:03d}/{:03d}] - Iteration [{:03d}/{:03d}]: tooth cd = {:.6f}, full cd = {:.6f}, color l1 = {:.6f}, total = {:.6f}"
                        .format(epoch, params.epochs, i + 1, len(train_dataloader), loss_tooth.item() * 1e3, loss_full.item() * 1e3, color_loss.item() * 1e3, loss.item() * 1e3))
                else:
                    log(log_fd, "Training Epoch [{:03d}/{:03d}] - Iteration [{:03d}/{:03d}]: tooth cd = {:.6f}, full cd = {:.6f}, total = {:.6f}"
                        .format(epoch, params.epochs, i + 1, len(train_dataloader), loss_tooth.item() * 1e3, loss_full.item() * 1e3, loss.item() * 1e3))

            train_writer.add_scalar('tooth_cd', loss_tooth.item(), train_step)
            train_writer.add_scalar('full_cd', loss_full.item(), train_step)
            if params.use_color:
                train_writer.add_scalar('color_l1', color_loss.item(), train_step)
            train_writer.add_scalar('total', loss.item(), train_step)
            train_step += 1

        lr_schedual.step()

        model.eval()
        total_cd_l1 = 0.0
        with torch.no_grad():
            rand_iter = random.randint(0, len(val_dataloader) - 1) if len(val_dataloader) > 0 else 0

            for i, batch in enumerate(val_dataloader):
                missing_pc = batch['missing_pc'].to(device)
                gt_tooth_pc = batch['gt_tooth_pc'].to(device)
                complete_pc = batch['complete_pc'].to(device)
                gt_tooth_color = batch['gt_tooth_color'].to(device) if params.use_color else None

                ret = model(missing_pc)
                pred_fine = model.get_pred_fine(ret)
                pred_color = model.get_pred_color(ret)

                loss_tooth = l1_cd(pred_fine, gt_tooth_pc)
                merged_pc = torch.cat([missing_pc, pred_fine], dim=1)
                loss_full = l1_cd(merged_pc, complete_pc)

                loss_val = params.local_weight * loss_tooth + params.full_weight * loss_full
                if params.use_color:
                    color_loss = symmetric_color_nn_l1(pred_fine, pred_color, gt_tooth_pc, gt_tooth_color)
                    loss_val = loss_val + params.color_weight * color_loss

                total_cd_l1 += loss_val.item()

                if rand_iter == i:
                    index = random.randint(0, pred_fine.shape[0] - 1)
                    plot_pcd_one_view(
                        os.path.join(epochs_dir, f'epoch_{epoch:03d}.jpg'),
                        [missing_pc[index].detach().cpu().numpy(), pred_fine[index].detach().cpu().numpy(), gt_tooth_pc[index].detach().cpu().numpy()],
                        ['Missing', 'Pred Tooth', 'GT Tooth'],
                        xlim=(-0.35, 0.35), ylim=(-0.35, 0.35), zlim=(-0.35, 0.35)
                    )

            if len(val_dataset) > 0:
                total_cd_l1 /= len(val_dataset)
            val_writer.add_scalar('l1_cd', total_cd_l1, val_step)
            val_step += 1

            log(log_fd, "Validate Epoch [{:03d}/{:03d}]: Total L1 CD = {:.6f}".format(epoch, params.epochs, total_cd_l1 * 1e3))

        torch.save(model.state_dict(), os.path.join(ckpt_dir, 'last_l1_cd.pth'))
        if total_cd_l1 < best_cd_l1:
            best_epoch_l1 = epoch
            best_cd_l1 = total_cd_l1
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_l1_cd.pth'))

    log(log_fd, f'Best l1 cd model in epoch {best_epoch_l1}, the minimum l1 cd is {best_cd_l1 * 1e3}')
    log_fd.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ToothNoMaskAdaPoinTr')
    parser.add_argument('--exp_name', type=str, default='arch2dc_stage1', help='Tag of experiment')
    parser.add_argument('--log_dir', type=str, default='logs/adapointr', help='Logger directory')
    parser.add_argument('--ckpt_path', type=str, default=None, help='The path of pretrained model')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='Epochs of training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for data loader')
    parser.add_argument('--num_workers', type=int, default=6, help='num_workers for data loader')
    parser.add_argument('--log_frequency', type=int, default=5, help='Logger frequency in every epoch')
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
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--input_num_points', type=int, default=4096, help='Total input points')
    parser.add_argument('--use_color', action='store_true', help='Enable RGB completion branch')
    parser.add_argument('--color_weight', type=float, default=0.2, help='Weight for color L1 loss')

    params = parser.parse_args()
    train(params)
