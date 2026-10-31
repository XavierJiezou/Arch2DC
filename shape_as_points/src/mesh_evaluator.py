# Mesh evaluation metrics (Chamfer-L1, F-score@1%, normal consistency).
#
# Adapted from NoKSR (https://github.com/theialab/noksr, noksr/utils/evaluation.py),
# licensed under the Apache License 2.0. Only the metrics used by Arch2DC are kept,
# and the dependency on `pycg` was removed. The numerical definitions are unchanged.
import logging

import numpy as np
import open3d as o3d
import torch
from pykdtree.kdtree import KDTree

NAN_METRIC = float("nan")


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    """每个源点到目标点云的最近距离，以及对应法向的 |cos| 一致性。"""
    kdtree = KDTree(points_tgt.astype(np.float32))
    dist, idx = kdtree.query(points_src.astype(np.float32))

    if normals_src is not None and normals_tgt is not None:
        normals_src = normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)
        # 法向方向不一致时取绝对值
        normals_dot_product = np.abs((normals_tgt[idx] * normals_src).sum(axis=-1))
    else:
        normals_dot_product = np.array([np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def get_threshold_percentage(dist, thresholds):
    return [(dist <= t).mean() for t in thresholds]


class MeshEvaluator:
    """返回字典含义：
    - chamfer-L1: 0.5 * (completeness + accuracy)
    - f-score:    阈值 0.01 的 F-score
    - normals:    双向法向一致性均值 (0-1)
    """

    ESSENTIAL_METRICS = ["chamfer-L1", "f-score", "normals"]

    def __init__(self, n_points=100000, metric_names=ESSENTIAL_METRICS):
        self.n_points = n_points
        self.thresholds = np.array([0.01, 0.015, 0.02, 0.002, 0.1])
        self.metric_names = metric_names

    def eval_mesh(self, mesh, pointcloud_tgt, normals_tgt):
        """mesh: o3d.geometry.TriangleMesh；pointcloud_tgt / normals_tgt: (N, 3) GT 点与法向。"""
        if isinstance(pointcloud_tgt, torch.Tensor):
            pointcloud_tgt = pointcloud_tgt.detach().cpu().numpy().astype(float)
        if isinstance(normals_tgt, torch.Tensor):
            normals_tgt = normals_tgt.detach().cpu().numpy().astype(float)

        try:
            # 固定随机种子保证可复现；使用三角面法向，与 SAP 一致
            o3d.utility.random.seed(0)
            sampled = mesh.sample_points_uniformly(number_of_points=self.n_points, use_triangle_normal=True)
            pointcloud = np.asarray(sampled.points)
            normals = np.asarray(sampled.normals)
        except RuntimeError:
            pointcloud = np.zeros((0, 3))
            normals = np.zeros((0, 3))
        return self._evaluate(pointcloud, pointcloud_tgt, normals, normals_tgt)

    def _evaluate(self, pointcloud, pointcloud_tgt, normals=None, normals_tgt=None):
        if pointcloud.shape[0] == 0:
            logging.warning("Empty pointcloud / mesh detected! Return NaN metric!")
            return {k: NAN_METRIC for k in self.metric_names}

        completeness, completeness_normals = distance_p2p(pointcloud_tgt, normals_tgt, pointcloud, normals)
        recall = get_threshold_percentage(completeness, self.thresholds)
        accuracy, accuracy_normals = distance_p2p(pointcloud, normals, pointcloud_tgt, normals_tgt)
        precision = get_threshold_percentage(accuracy, self.thresholds)

        completeness2 = (completeness ** 2).mean()
        accuracy2 = (accuracy ** 2).mean()
        completeness = completeness.mean()
        accuracy = accuracy.mean()

        f = [2 * precision[i] * recall[i] / (precision[i] + recall[i]) for i in range(len(precision))]
        return {
            "completeness": completeness,
            "accuracy": accuracy,
            "chamfer-L2": 0.5 * (completeness2 + accuracy2),
            "chamfer-L1": 0.5 * (completeness + accuracy),
            "f-score": f[0],  # 阈值 0.01
            "f-score-15": f[1],
            "f-score-20": f[2],
            "normals": 0.5 * completeness_normals.mean() + 0.5 * accuracy_normals.mean(),
        }
