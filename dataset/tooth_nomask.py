"""
牙齿点云补全数据集 - 无Mask版本

数据格式:
- 0号文件: 完整的点云（全部牙齿）
- 1号文件: 缺失某一颗牙齿的点云
- 2号文件: 缺失半颗牙齿的点云（本版本暂不使用）
- 3号文件: 缺失处完整牙齿的GT
- 4号文件: 缺失处半颗牙齿的GT（本版本暂不使用）

流程:
1. 读取0号和1号文件，对齐后进行归一化
2. 模型输入: 1号文件（缺损上颌）
3. 模型输出: 预测的缺失牙齿点云
4. 监督信号: 3号文件（缺失处完整牙齿GT）
5. 附加损失: 预测牙齿嵌入回1号文件后与0号文件的CD损失
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
import open3d as o3d


class ToothNoMask(Dataset):
    def __init__(
        self, 
        dataroot="data/tooth2", 
        split="train", 
        missing_num_points=8192,      # 缺损上颌点云的采样数量
        gt_tooth_num_points=2048,     # GT单颗牙齿点云的采样数量
        complete_num_points=16384,    # 完整点云的采样数量
        train_ratio=0.7,              # 训练集比例
        val_ratio=0.2,                # 验证集比例
        test_ratio=0.1,               # 测试集比例
        return_normals=False,         # 是否返回GT牙齿法向（用于联合表面重建）
        normal_k=30,                  # 点云法向估计KNN
        return_colors=False,          # 是否返回颜色（用于颜色补全）
    ):
        """
        Args:
            dataroot: 数据根目录
            split: 'train' 或 'val'
            missing_num_points: 缺损上颌点云的采样数量（1号文件）
            gt_tooth_num_points: GT单颗牙齿点云的采样数量（3号文件）
            complete_num_points: 完整点云的采样数量（0号文件）
            train_ratio: 训练集比例
        """
        self.missing_num_points = missing_num_points
        self.gt_tooth_num_points = gt_tooth_num_points
        self.complete_num_points = complete_num_points
        self.return_normals = return_normals
        self.normal_k = normal_k
        self.return_colors = return_colors
        
        self.data_dir = dataroot
        
        # 获取所有数据样本 (以_0.ply结尾的文件)
        all_files = [f for f in os.listdir(self.data_dir) if f.endswith('_0.ply')]
        all_files = sorted(all_files, key=lambda x: int(x.split('_')[0]))
        
        # 根据split划分数据集
        n_total = len(all_files)
        ratio_sum = train_ratio + val_ratio + test_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum}")

        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        if split == 'train':
            self.files = all_files[:n_train]
        elif split == 'val':
            self.files = all_files[n_train:n_train + n_val]
        elif split == 'test':
            self.files = all_files[n_train + n_val:]
        else:
            raise ValueError(f"Unsupported split: {split}")
            
        print(f"ToothNoMask Dataset [{split}]: {len(self.files)} samples loaded")
    
    def __len__(self):
        return len(self.files)
    
    def pc_norm(self, pc, center=None, scale=None):
        """
        点云归一化
        Args:
            pc: NxC 点云
            center: 如果提供，使用该中心进行归一化
            scale: 如果提供，使用该尺度进行归一化
        Returns:
            归一化后的点云，中心，尺度
        """
        if center is None:
            center = np.mean(pc, axis=0)
        pc_centered = pc - center
        
        if scale is None:
            scale = np.max(np.sqrt(np.sum(pc_centered**2, axis=1)))
        
        pc_normalized = pc_centered / scale
        return pc_normalized, center, scale
    
    def sample_points(self, points, num_points, replace=False):
        """
        对点云进行采样
        """
        n = points.shape[0]
        if n >= num_points:
            indices = np.random.choice(n, num_points, replace=False)
        else:
            indices = np.random.choice(n, num_points, replace=True)
        return points[indices]

    def estimate_normals(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=self.normal_k))
        normals = np.asarray(pcd.normals).astype(np.float32)
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        return normals

    def get_or_default_colors(self, pcd):
        colors = np.asarray(pcd.colors)
        points = np.asarray(pcd.points)
        if colors.shape != points.shape or colors.shape[0] == 0:
            return np.zeros((points.shape[0], 3), dtype=np.float32)
        colors = colors.astype(np.float32)
        return np.clip(colors, 0.0, 1.0)

    def sample_points_with_idx(self, points, num_points):
        n = points.shape[0]
        replace = n < num_points
        idx = np.random.choice(n, num_points, replace=replace)
        return points[idx], idx
    
    def __getitem__(self, idx):
        """
        Returns:
            missing_pc: 缺损上颌点云 (1号文件) - 模型输入之一
            gt_tooth_pc: GT单颗牙齿点云 (3号文件) - 监督信号
            complete_pc: 完整点云 (0号文件) - 用于计算嵌入后的损失
        """
        base_file = self.files[idx]
        base_name = base_file.replace('_0.ply', '')
        
        # 构建文件路径
        complete_path = os.path.join(self.data_dir, f"{base_name}_0.ply")
        missing_path = os.path.join(self.data_dir, f"{base_name}_1.ply")
        gt_tooth_path = os.path.join(self.data_dir, f"{base_name}_3.ply")
        
        # 读取点云
        complete_pc = o3d.io.read_point_cloud(complete_path)
        missing_pc = o3d.io.read_point_cloud(missing_path)
        gt_tooth_pc = o3d.io.read_point_cloud(gt_tooth_path)
        
        complete_points = np.asarray(complete_pc.points)
        missing_points = np.asarray(missing_pc.points)
        gt_tooth_points = np.asarray(gt_tooth_pc.points)

        complete_colors = None
        missing_colors = None
        gt_tooth_colors = None
        if self.return_colors:
            complete_colors = self.get_or_default_colors(complete_pc)
            missing_colors = self.get_or_default_colors(missing_pc)
            gt_tooth_colors = self.get_or_default_colors(gt_tooth_pc)

        gt_tooth_normals = None
        if self.return_normals:
            gt_mesh = o3d.io.read_triangle_mesh(gt_tooth_path)
            if gt_mesh.has_triangles() and len(gt_mesh.triangles) > 0:
                gt_mesh.compute_vertex_normals()
                sampled = gt_mesh.sample_points_uniformly(number_of_points=self.gt_tooth_num_points)
                gt_tooth_points = np.asarray(sampled.points)
                gt_tooth_normals = np.asarray(sampled.normals)
            else:
                gt_tooth_normals = self.estimate_normals(gt_tooth_points)
        
        # 使用完整点云的统计量进行归一化（保证所有点云在同一坐标系）
        complete_points_norm, center, scale = self.pc_norm(complete_points)
        missing_points_norm, _, _ = self.pc_norm(missing_points, center, scale)
        gt_tooth_points_norm, _, _ = self.pc_norm(gt_tooth_points, center, scale)
        
        # 采样
        complete_sampled = self.sample_points(complete_points_norm, self.complete_num_points)
        missing_sampled = self.sample_points(missing_points_norm, self.missing_num_points)

        if self.return_colors:
            complete_sampled, complete_idx = self.sample_points_with_idx(complete_points_norm, self.complete_num_points)
            complete_colors_sampled = complete_colors[complete_idx]

            missing_sampled, missing_idx = self.sample_points_with_idx(missing_points_norm, self.missing_num_points)
            missing_colors_sampled = missing_colors[missing_idx]

        if self.return_normals:
            n = gt_tooth_points_norm.shape[0]
            replace = n < self.gt_tooth_num_points
            idxs = np.random.choice(n, self.gt_tooth_num_points, replace=replace)
            gt_tooth_sampled = gt_tooth_points_norm[idxs]
            gt_tooth_normals = gt_tooth_normals[idxs].astype(np.float32)
            gt_tooth_normals = gt_tooth_normals / (np.linalg.norm(gt_tooth_normals, axis=1, keepdims=True) + 1e-8)

            ret = {
                'missing_pc': torch.tensor(missing_sampled, dtype=torch.float32),
                'gt_tooth_pc': torch.tensor(gt_tooth_sampled, dtype=torch.float32),
                'gt_tooth_normals': torch.tensor(gt_tooth_normals, dtype=torch.float32),
                'complete_pc': torch.tensor(complete_sampled, dtype=torch.float32),
                'norm_center': torch.tensor(center, dtype=torch.float32),
                'norm_scale': torch.tensor(scale, dtype=torch.float32),
            }

            if self.return_colors:
                gt_tooth_colors_sampled = gt_tooth_colors[idxs].astype(np.float32)
                ret.update({
                    'missing_color': torch.tensor(missing_colors_sampled, dtype=torch.float32),
                    'gt_tooth_color': torch.tensor(gt_tooth_colors_sampled, dtype=torch.float32),
                    'complete_color': torch.tensor(complete_colors_sampled, dtype=torch.float32),
                })

            return ret

        gt_tooth_sampled = self.sample_points(gt_tooth_points_norm, self.gt_tooth_num_points)
        ret = {
            'missing_pc': torch.tensor(missing_sampled, dtype=torch.float32),
            'gt_tooth_pc': torch.tensor(gt_tooth_sampled, dtype=torch.float32),
            'complete_pc': torch.tensor(complete_sampled, dtype=torch.float32),
            'norm_center': torch.tensor(center, dtype=torch.float32),
            'norm_scale': torch.tensor(scale, dtype=torch.float32),
        }

        if self.return_colors:
            gt_tooth_sampled, gt_idx = self.sample_points_with_idx(gt_tooth_points_norm, self.gt_tooth_num_points)
            gt_tooth_colors_sampled = gt_tooth_colors[gt_idx].astype(np.float32)
            ret = {
                'missing_pc': torch.tensor(missing_sampled, dtype=torch.float32),
                'gt_tooth_pc': torch.tensor(gt_tooth_sampled, dtype=torch.float32),
                'complete_pc': torch.tensor(complete_sampled, dtype=torch.float32),
                'missing_color': torch.tensor(missing_colors_sampled, dtype=torch.float32),
                'gt_tooth_color': torch.tensor(gt_tooth_colors_sampled, dtype=torch.float32),
                'complete_color': torch.tensor(complete_colors_sampled, dtype=torch.float32),
                'norm_center': torch.tensor(center, dtype=torch.float32),
                'norm_scale': torch.tensor(scale, dtype=torch.float32),
            }

        return ret


class ToothNoMaskCollate:
    """自定义collate函数，用于处理不同大小的点云"""
    def __call__(self, batch):
        missing_pcs = torch.stack([item['missing_pc'] for item in batch])
        gt_tooth_pcs = torch.stack([item['gt_tooth_pc'] for item in batch])
        complete_pcs = torch.stack([item['complete_pc'] for item in batch])

        ret = {
            'missing_pc': missing_pcs,
            'gt_tooth_pc': gt_tooth_pcs,
            'complete_pc': complete_pcs,
        }

        if 'missing_color' in batch[0]:
            ret['missing_color'] = torch.stack([item['missing_color'] for item in batch])
            ret['gt_tooth_color'] = torch.stack([item['gt_tooth_color'] for item in batch])
            ret['complete_color'] = torch.stack([item['complete_color'] for item in batch])

        return ret


if __name__ == '__main__':
    # 测试数据集
    dataset = ToothNoMask('data/tooth2', 'train')
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Missing PC shape: {sample['missing_pc'].shape}")
    print(f"GT Tooth PC shape: {sample['gt_tooth_pc'].shape}")
    print(f"Complete PC shape: {sample['complete_pc'].shape}")
    
    # 检查数据范围
    print(f"Missing PC range: [{sample['missing_pc'].min():.4f}, {sample['missing_pc'].max():.4f}]")
