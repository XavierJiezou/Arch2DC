import hashlib
import os

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils import data
import trimesh

from src.dpsr import DPSR


def _as_mesh_or_cloud(path):
    geom = trimesh.load(path, process=False)
    if isinstance(geom, trimesh.Scene):
        geoms = [g for g in geom.geometry.values() if hasattr(g, "vertices")]
        if not geoms:
            raise ValueError(f"No geometry with vertices in {path}")
        geom = trimesh.util.concatenate(geoms)
    if not hasattr(geom, "vertices") or len(geom.vertices) == 0:
        raise ValueError(f"Empty geometry: {path}")
    return geom


def _estimate_normals(points, k=30):
    points = np.asarray(points, dtype=np.float32)
    n_points = len(points)
    if n_points == 0:
        return np.zeros((0, 3), dtype=np.float32)
    tree = cKDTree(points)
    _, indices = tree.query(points, k=min(k, n_points))
    if indices.ndim == 1:
        indices = indices[:, None]
    normals = np.zeros_like(points, dtype=np.float32)
    center = points.mean(axis=0)
    for i in range(n_points):
        neigh = points[indices[i]]
        cov = np.cov(neigh.T)
        _, vecs = np.linalg.eigh(cov)
        normal = vecs[:, 0]
        if np.dot(normal, points[i] - center) < 0:
            normal = -normal
        normals[i] = normal / (np.linalg.norm(normal) + 1.0e-8)
    return normals


def _load_vertices_and_normals(path):
    geom = _as_mesh_or_cloud(path)
    points = np.asarray(geom.vertices, dtype=np.float32)
    normals = None
    if hasattr(geom, "vertex_normals") and len(geom.vertex_normals) == len(points):
        normals = np.asarray(geom.vertex_normals, dtype=np.float32)
    if normals is None or len(normals) != len(points):
        normals = _estimate_normals(points)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1.0e-8)
    return points, normals.astype(np.float32)


def _sample_surface(path, n_points):
    geom = _as_mesh_or_cloud(path)
    if hasattr(geom, "faces") and len(getattr(geom, "faces", [])) > 0:
        points, face_ids = trimesh.sample.sample_surface(geom, n_points)
        normals = np.asarray(geom.face_normals[face_ids], dtype=np.float32)
    else:
        vertices, vertex_normals = _load_vertices_and_normals(path)
        ids = np.random.choice(len(vertices), n_points, replace=len(vertices) < n_points)
        points = vertices[ids]
        normals = vertex_normals[ids]
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1.0e-8)
    return points.astype(np.float32), normals.astype(np.float32)


def _load_vertices(path):
    """只读顶点坐标（不估计法向，速度快），用于计算归一化包围盒。"""
    return np.asarray(_as_mesh_or_cloud(path).vertices, dtype=np.float32)


def missing_arch_file(file0_path):
    """由完整牙弓路径 xxx_0.ply 推出缺损牙弓路径 xxx_1.ply。"""
    path = str(file0_path)
    if not path.endswith("_0.ply"):
        raise ValueError(f"Cannot derive X1 path from {file0_path}")
    x1 = path[: -len("_0.ply")] + "_1.ply"
    if not os.path.exists(x1):
        raise FileNotFoundError(f"Missing arch file not found: {x1}")
    return x1


def bbox_normalizer(ref_points):
    """由参考点云的轴对齐包围盒得到 SAP 映射 p -> (p - center) / scale + 0.5 的参数。"""
    ref_points = np.asarray(ref_points, dtype=np.float32)
    p_min = ref_points.min(axis=0)
    p_max = ref_points.max(axis=0)
    center = ((p_min + p_max) * 0.5).astype(np.float32)
    scale = float(np.max(p_max - p_min))
    if scale <= 1.0e-8:
        raise ValueError("Degenerate bbox for normalization")
    return center, np.float32(scale)


def sap_normalizer(x1_path, pred_points):
    """SAP 坐标归一化：缺损牙弓 X1 ∪ Stage I 预测牙冠 的包围盒（二者推理时均可得）。
    并上预测牙冠可保证缺失牙位于牙弓末端时目标区域仍落在网格内。"""
    ref = np.concatenate([_load_vertices(x1_path), np.asarray(pred_points, dtype=np.float32)], axis=0)
    return bbox_normalizer(ref)


def _sample_fixed(points, normals, n_points):
    ids = np.random.choice(len(points), n_points, replace=len(points) < n_points)
    return points[ids].astype(np.float32), normals[ids].astype(np.float32)


class ToothDualDataset(data.Dataset):
    """Three-column predtooth/file0/file3 dataset for SAP dual supervision."""

    def __init__(
        self,
        list_file,
        pointcloud_n=3000,
        num_gt_points=10000,
        global_num_points=10000,
        file0_surface_points=50000,
        global_dist_threshold=0.05,
        global_min_points=2048,
        grid_res=128,
        psr_sigma=2,
        cache_dir=None,
        split="train",
    ):
        self.list_file = list_file
        self.pointcloud_n = int(pointcloud_n)
        self.num_gt_points = int(num_gt_points)
        self.global_num_points = int(global_num_points)
        self.file0_surface_points = int(file0_surface_points)
        self.global_dist_threshold = float(global_dist_threshold)
        self.global_min_points = int(global_min_points)
        self.grid_res = int(grid_res)
        self.psr_sigma = psr_sigma
        self.cache_dir = cache_dir
        self.split = split

        with open(list_file) as f:
            self.entries = [line.strip().split()[:3] for line in f if line.strip()]
        self.entries = [e for e in self.entries if len(e) == 3]
        if not self.entries:
            raise ValueError(f"No valid three-column entries in {list_file}")

        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.entries)

    def _normalizer(self, file0_path, pred_points):
        center, scale = sap_normalizer(missing_arch_file(file0_path), pred_points)

        def norm(points):
            return (points - center[None, :]) / scale + 0.5

        return center.astype(np.float32), np.float32(scale), norm

    def _psr_cache_path(self, file3_path, center, scale):
        if self.cache_dir is None:
            return None
        norm_key = ",".join([f"{x:.8f}" for x in center.tolist()] + [f"{float(scale):.8f}"])
        digest = hashlib.md5(
            f"{file3_path}|{norm_key}|{self.num_gt_points}|{self.grid_res}|{self.psr_sigma}".encode()
        ).hexdigest()
        return os.path.join(self.cache_dir, f"{digest}.npz")

    def _load_or_make_local_psr(self, file3_path, center, scale, points, normals):
        cache_path = self._psr_cache_path(file3_path, center, scale)
        if cache_path is not None and os.path.exists(cache_path):
            return np.load(cache_path)["psr"].astype(np.float32)

        with torch.no_grad():
            dpsr = DPSR(res=(self.grid_res, self.grid_res, self.grid_res), sig=self.psr_sigma)
            pts_t = torch.from_numpy(points[None]).float()
            nrm_t = torch.from_numpy(normals[None]).float()
            psr = dpsr(pts_t, nrm_t)[0].cpu().numpy().astype(np.float32)

        if cache_path is not None:
            np.savez_compressed(cache_path, psr=psr)
        return psr

    def _global_near_points(self, file0_path, local_points_norm, norm):
        file0_points, file0_normals = _sample_surface(file0_path, self.file0_surface_points)
        file0_points = norm(file0_points).astype(np.float32)
        tree = cKDTree(local_points_norm)

        threshold = self.global_dist_threshold
        selected = None
        for _ in range(4):
            dists, _ = tree.query(file0_points, k=1)
            mask = dists < threshold
            if mask.sum() >= self.global_min_points:
                selected = mask
                break
            threshold *= 2.0
        if selected is None:
            selected = mask
        if selected.sum() == 0:
            selected = np.ones(len(file0_points), dtype=bool)

        points = file0_points[selected]
        normals = file0_normals[selected]
        return _sample_fixed(points, normals, self.global_num_points)

    def __getitem__(self, idx):
        pred_path, file0_path, file3_path = self.entries[idx]
        pred_points, pred_normals = _load_vertices_and_normals(pred_path)
        center, scale, norm = self._normalizer(file0_path, pred_points)

        pred_points = np.clip(norm(pred_points), 0.0, 0.99).astype(np.float32)
        pred_points, pred_normals = _sample_fixed(pred_points, pred_normals, self.pointcloud_n)

        local_points_raw, local_normals = _sample_surface(file3_path, self.num_gt_points)
        local_points = np.clip(norm(local_points_raw), 0.0, 0.99).astype(np.float32)
        local_psr = self._load_or_make_local_psr(file3_path, center, scale, local_points, local_normals)

        global_points, global_normals = self._global_near_points(file0_path, local_points, norm)
        global_points = np.clip(global_points, 0.0, 0.99).astype(np.float32)

        scene_name = os.path.basename(os.path.dirname(pred_path))
        stem = os.path.splitext(os.path.basename(pred_path))[0]

        return {
            "inputs": pred_points,
            "inputs.normals": pred_normals.astype(np.float32),
            "gt_points_local": local_points,
            "gt_normals_local": local_normals.astype(np.float32),
            "gt_psr_local": local_psr,
            "gt_points_global": global_points,
            "gt_normals_global": global_normals.astype(np.float32),
            "norm_center": center,
            "norm_scale": np.asarray([scale], dtype=np.float32),
            "model_id": f"{scene_name}_{stem}",
        }
