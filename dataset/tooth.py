import os
import torch
import numpy as np
from torch.utils.data import Dataset
import open3d as o3d


class Tooth(Dataset):
    def __init__(self, dataroot="data/tooth", split="train", p_num_points=2048, c_num_points=16384, half=True):
        # p_num_points： 缺失牙齿点云的数量（作为模型的输入）
        # c_num_points： 完整牙齿点云的数量（作为Groud Truth：真值）
        self.p_num_points = p_num_points
        self.c_num_points = c_num_points
        self.data_dir = os.path.join(dataroot, split)
        self.files = [f for f in os.listdir(self.data_dir) if f.endswith('_0.ply')]
        self.half = half
    def __len__(self):
        return len(self.files)
    
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        
        
        # change scale
        # data=pc
        # xyz_min = np.min(data[:,0:3],axis=0)
        # xyz_max = np.max(data[:,0:3],axis=0)
        # xyz_move = xyz_min+(xyz_max-xyz_min)/2
        # data[:,0:3] = data[:,0:3]-xyz_move
        
        # #scale
        # scale = np.max(data[:,0:3])
        # data[:,0:3] = data[:,0:3]/scale
        # return data
    
    def __getitem__(self, idx):
        original_file = self.files[idx]
        if self.half:
            missing_file = original_file.replace('_0.ply', '_2.ply')
        else:
            missing_file = original_file.replace('_0.ply', '_1.ply')

        # 读取原始和缺失部分的点云
        original_path = os.path.join(self.data_dir, original_file)
        missing_path = os.path.join(self.data_dir, missing_file)
        
        original_pc = o3d.io.read_point_cloud(original_path)
        missing_pc = o3d.io.read_point_cloud(missing_path)

        original_points = np.asarray(original_pc.points)
        missing_points = np.asarray(missing_pc.points)
        
        original_points = self.pc_norm(original_points)
        missing_points = self.pc_norm(missing_points)

        # 保证点数一致
        if original_points.shape[0] > self.c_num_points:
            original_points = original_points[np.random.choice(original_points.shape[0], self.c_num_points, replace=False)]
        if missing_points.shape[0] > self.p_num_points:
            missing_points = missing_points[np.random.choice(missing_points.shape[0], self.p_num_points, replace=False)]
        
        return torch.tensor(missing_points, dtype=torch.float32), torch.tensor(original_points, dtype=torch.float32)


if __name__=='__main__':
    train_dataset = Tooth('data/tooth', 'train')
    for p, c in train_dataset:
        print(p.shape, c.shape)
        print(p.max(), p.min())
        # break