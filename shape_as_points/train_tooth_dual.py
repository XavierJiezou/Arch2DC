import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import argparse
import shutil
import time

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.data.tooth_dual import ToothDualDataset
from src.data import collate_remove_none, worker_init_fn
from src.model import Encode2Points
from src.training_tooth_dual import ToothDualTrainer
from src.utils import AverageMeter, initialize_logger, load_config, load_model_manual


def build_dataset(cfg, split):
    list_key = f"{split}_list"
    list_file = cfg["data"].get(list_key)
    if list_file is None:
        raise ValueError(f"Missing data:{list_key} in config or command override.")
    return ToothDualDataset(
        list_file=list_file,
        pointcloud_n=cfg["data"]["pointcloud_n"],
        num_gt_points=cfg["data"]["num_gt_points"],
        global_num_points=cfg["data"]["global_num_points"],
        file0_surface_points=cfg["data"]["file0_surface_points"],
        global_dist_threshold=cfg["data"]["global_dist_threshold"],
        global_min_points=cfg["data"]["global_min_points"],
        grid_res=cfg["model"]["grid_res"],
        psr_sigma=cfg["model"]["psr_sigma"],
        cache_dir=cfg["data"].get("psr_cache_dir"),
        split=split,
    )


def main():
    parser = argparse.ArgumentParser(description="Train SAP tooth dual supervision.")
    parser.add_argument("config", type=str, help="Path to config file.")
    parser.add_argument("--train_list", type=str, default=None)
    parser.add_argument("--val_list", type=str, default=None)
    parser.add_argument("--no_cuda", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config, "configs/default.yaml")
    if args.train_list is not None:
        cfg["data"]["train_list"] = args.train_list
    if args.val_list is not None:
        cfg["data"]["val_list"] = args.val_list

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if cfg["train"]["timestamp"]:
        cfg["train"]["out_dir"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    logger = initialize_logger(cfg)
    if use_cuda:
        logger.info("using GPU: %s", torch.cuda.get_device_name(0))
    else:
        logger.info("using CPU")
    with open(os.path.join(cfg["train"]["out_dir"], "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    writer = SummaryWriter(log_dir=os.path.join(cfg["train"]["out_dir"], "tensorboard_log"))

    train_dataset = build_dataset(cfg, "train")
    val_dataset = build_dataset(cfg, "val")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["n_workers"],
        shuffle=True,
        collate_fn=collate_remove_none,
        worker_init_fn=worker_init_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=cfg["train"]["n_workers_val"],
        shuffle=False,
        collate_fn=collate_remove_none,
        worker_init_fn=worker_init_fn,
    )

    if torch.cuda.device_count() > 1 and use_cuda:
        model = torch.nn.DataParallel(Encode2Points(cfg)).to(device)
    else:
        model = Encode2Points(cfg).to(device)

    optimizer = optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))
    trainer = ToothDualTrainer(cfg, optimizer, device=device)

    checkpoint_path = os.path.join(cfg["train"]["out_dir"], "model.pt")
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        load_model_manual(state_dict["state_dict"], model)
        optimizer.load_state_dict(state_dict["optimizer"])
        start_epoch = state_dict.get("epoch", -1)
        it = state_dict.get("it", -1)
        metric_val_best = state_dict.get("loss_val_best", np.inf)
        logger.info("Load model from iteration %d", it)
    except Exception:
        start_epoch = -1
        it = -1
        metric_val_best = np.inf

    runtime = AverageMeter()
    metric_name = cfg["train"]["model_selection_metric"]

    def save_state(path, epoch, it, metric_val_best):
        state = {
            "epoch": epoch,
            "it": it,
            "loss_val_best": metric_val_best,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(state, path)

    def run_validation(epoch, it, metric_val_best):
        eval_dict = trainer.evaluate(val_loader, model)
        metric_val = eval_dict[metric_name]
        logger.info("Validation metric (%s): %.6f", metric_name, metric_val)
        for key, value in eval_dict.items():
            writer.add_scalar(f"val/{key}", value, it)
        if metric_val <= metric_val_best:
            metric_val_best = metric_val
            save_state(os.path.join(cfg["train"]["out_dir"], "model_best.pt"), epoch, it, metric_val_best)
            logger.info("New best model (loss %.6f)", metric_val_best)
        model.train()
        return metric_val_best

    for epoch in range(start_epoch + 1, cfg["train"]["total_epochs"] + 1):
        model.train()
        for batch in train_loader:
            it += 1
            start = time.time()
            loss, loss_each = trainer.train_step(batch, model)
            runtime.update(time.time() - start)

            if it % cfg["train"]["print_every"] == 0:
                writer.add_scalar("train/loss", loss, it)
                parts = [f"[Epoch {epoch:02d}] it={it} loss={loss:.4f}"]
                for key, value in loss_each.items():
                    writer.add_scalar(f"train/{key}", value, it)
                    parts.append(f"{key}={value.item():.4f}")
                parts.append(f"time={runtime.val:.3f}/{runtime.sum:.2f}")
                logger.info(" ".join(parts))

            if it > 0 and it % cfg["train"]["validate_every"] == 0:
                metric_val_best = run_validation(epoch, it, metric_val_best)

            if epoch > 0 and it % cfg["train"]["checkpoint_every"] == 0:
                save_state(checkpoint_path, epoch, it, metric_val_best)
                if it % cfg["train"]["backup_every"] == 0:
                    save_state(os.path.join(cfg["train"]["dir_model"], f"{it:04d}.pt"), epoch, it, metric_val_best)
                logger.info("Save checkpoint at iteration %d", it)

        validate_every_epoch = cfg["train"].get("validate_every_epoch", 0)
        if validate_every_epoch and epoch > 0 and epoch % validate_every_epoch == 0:
            metric_val_best = run_validation(epoch, it, metric_val_best)
            save_state(checkpoint_path, epoch, it, metric_val_best)
            logger.info("Save checkpoint at epoch %d iteration %d", epoch, it)


if __name__ == "__main__":
    main()
