from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from pytorch3d.ops import knn_gather, knn_points

from src.dpsr import DPSR


class ToothDualTrainer:
    def __init__(self, cfg, optimizer=None, device=None):
        self.cfg = cfg
        self.optimizer = optimizer
        self.device = device
        self.dpsr = DPSR(
            res=(cfg["model"]["grid_res"], cfg["model"]["grid_res"], cfg["model"]["grid_res"]),
            sig=cfg["model"]["psr_sigma"],
        ).to(device)

    def _normal_loss(self, pred_points, pred_normals, gt_points, gt_normals):
        knn = knn_points(pred_points, gt_points, K=1)
        nearest_normals = knn_gather(gt_normals, knn.idx)[..., 0, :]
        return F.l1_loss(pred_normals, nearest_normals)

    def _one_sided_chamfer(self, src, tgt):
        return knn_points(src, tgt, K=1).dists[..., 0].mean()

    def _losses(self, batch, model):
        inputs = batch["inputs"].to(self.device)
        points, normals = model(inputs)

        gt_psr_local = batch["gt_psr_local"].to(self.device)
        gt_points_local = batch["gt_points_local"].to(self.device)
        gt_normals_local = batch["gt_normals_local"].to(self.device)
        gt_points_global = batch["gt_points_global"].to(self.device)
        gt_normals_global = batch["gt_normals_global"].to(self.device)

        psr_pred = self.dpsr(points, normals)
        if self.cfg["model"].get("psr_tanh", False):
            psr_pred = torch.tanh(psr_pred)
            gt_psr_local = torch.tanh(gt_psr_local)

        weights = self.cfg["train"]
        local_psr = F.mse_loss(psr_pred, gt_psr_local)
        local_cd = self._one_sided_chamfer(points, gt_points_local)
        local_normals = self._normal_loss(points, normals, gt_points_local, gt_normals_local)
        global_cd = self._one_sided_chamfer(points, gt_points_global)
        global_normals = self._normal_loss(points, normals, gt_points_global, gt_normals_global)

        loss_each = {
            "local_psr": weights["w_psr_local"] * local_psr,
            "local_cd": weights["w_cd_local"] * local_cd,
            "local_normals": weights["w_normal_local"] * local_normals,
            "global_cd": weights["w_global"] * weights["w_cd_global"] * global_cd,
            "global_normals": weights["w_global"] * weights["w_normal_global"] * global_normals,
        }
        total = sum(loss_each.values())
        return total, loss_each, points, normals

    def train_step(self, batch, model):
        self.optimizer.zero_grad()
        loss, loss_each, _, _ = self._losses(batch, model)
        loss.backward()
        self.optimizer.step()
        return loss.item(), {k: v.detach() for k, v in loss_each.items()}

    @torch.no_grad()
    def eval_step(self, batch, model):
        model.eval()
        loss, loss_each, _, _ = self._losses(batch, model)
        out = {"loss": loss.item()}
        out.update({k: v.item() for k, v in loss_each.items()})
        return out

    @torch.no_grad()
    def evaluate(self, loader, model):
        values = defaultdict(list)
        for batch in tqdm(loader, ncols=100):
            result = self.eval_step(batch, model)
            for key, value in result.items():
                values[key].append(value)
        return {key: float(np.mean(value)) for key, value in values.items()}
