"""
牙齿点云补全模型（无Mask输入）

输入:
- missing_pc: 缺失某颗牙齿的上颌点云

输出:
- 预测的缺失单颗牙齿点云
"""

import torch
import torch.nn as nn

from models.adapointr import AdaPoinTr


class ToothNoMaskAdaPoinTr(nn.Module):
    def __init__(self, config, input_num_points=None, enable_color=False, **kwargs):
        super().__init__()
        self.base_model = AdaPoinTr(config, **kwargs)
        self.input_num_points = input_num_points
        self.enable_color = enable_color
        if self.enable_color:
            # Predict RGB from reconstructed tooth geometry.
            self.color_head = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 3),
                nn.Sigmoid(),
            )

    def _sample_points(self, points, num_points):
        """Randomly sample points to a fixed size.
        Args:
            points: (B, N, 3)
            num_points: int
        Returns:
            (B, num_points, 3)
        """
        if num_points is None:
            return points
        if points.size(1) <= num_points:
            return points

        b, n, _ = points.shape
        idx_list = []
        for _ in range(b):
            idx = torch.randperm(n, device=points.device)[:num_points]
            idx_list.append(idx)
        idx = torch.stack(idx_list, dim=0)
        return torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))

    def forward(self, missing_pc):
        """
        Args:
            missing_pc: (B, N1, 3)
        Returns:
            与AdaPoinTr一致的输出
        """
        input_pc = missing_pc
        input_pc = self._sample_points(input_pc, self.input_num_points)
        geom_ret = self.base_model(input_pc)
        if not self.enable_color:
            return geom_ret

        pred_fine = geom_ret[-1]
        pred_color = self.color_head(pred_fine)
        return {
            'geom': geom_ret,
            'pred_color': pred_color,
        }

    def get_loss(self, ret, gt, epoch=1):
        """代理到内部 AdaPoinTr 的 get_loss"""
        return self.base_model.get_loss(ret, gt, epoch)

    def get_pred_fine(self, ret):
        """
        从模型输出中提取细粒度预测
        """
        if isinstance(ret, dict):
            ret = ret['geom']

        if self.training:
            # training输出: (pred_coarse, denoised_coarse, denoised_fine, pred_fine)
            return ret[-1]
        # eval输出: (coarse_point_cloud, rebuild_points)
        return ret[-1]

    def get_pred_color(self, ret):
        if isinstance(ret, dict):
            return ret['pred_color']
        return None
