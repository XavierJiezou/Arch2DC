import torch
from torch import nn
import open3d as o3d
from scipy.spatial.distance import directed_hausdorff

from extensions.chamfer_distance.chamfer_distance import ChamferDistance
from extensions.earth_movers_distance.emd import EarthMoverDistance


CD = ChamferDistance()
EMD = EarthMoverDistance()


def l2_cd(pcs1, pcs2):
    dist1, dist2 = CD(pcs1, pcs2)
    dist1 = torch.mean(dist1, dim=1)
    dist2 = torch.mean(dist2, dim=1)
    return torch.sum(dist1 + dist2)


def l1_cd(pcs1, pcs2):
    dist1, dist2 = CD(pcs1, pcs2)
    dist1 = torch.mean(torch.sqrt(dist1), 1)
    dist2 = torch.mean(torch.sqrt(dist2), 1)
    return torch.sum(dist1 + dist2) / 2


def emd(pcs1, pcs2):
    dists = EMD(pcs1, pcs2)
    return torch.sum(dists)


def f_score(pred, gt, th=0.01):
    """
    References: https://github.com/lmb-freiburg/what3d/blob/master/util.py

    Args:
        pred (np.ndarray): (N1, 3)
        gt   (np.ndarray): (N2, 3)
        th   (float): a distance threshhold
    """
    pred = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred))
    gt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt))

    dist1 = pred.compute_point_cloud_distance(gt)
    dist2 = gt.compute_point_cloud_distance(pred)

    recall = float(sum(d < th for d in dist2)) / float(len(dist2))
    precision = float(sum(d < th for d in dist1)) / float(len(dist1))
    return 2 * recall * precision / (recall + precision) if recall + precision else 0


def hausdorff_distance(pc1, pc2):
    """
    计算双向 Hausdorff 距离，返回最大双向距离。
    :param pc1: numpy array, shape (N, 3)
    :param pc2: numpy array, shape (M, 3)
    """
    forward_hd = directed_hausdorff(pc1, pc2)[0]
    backward_hd = directed_hausdorff(pc2, pc1)[0]
    return max(forward_hd, backward_hd)

# for adapointr
class ChamferDistanceL1(nn.Module):
    """
    L1 Chamfer Distance: sqrt(dist), mean over each sample, then mean over batch.
    Equivalent to the l1_cd() function.
    """
    def __init__(self):
        super(ChamferDistanceL1, self).__init__()
        self.cd = ChamferDistance()

    def forward(self, pcs1, pcs2):
        """
        Args:
            pcs1: Tensor (B, N, 3)
            pcs2: Tensor (B, M, 3)
        Returns:
            Scalar tensor: average L1 CD across batch
        """
        dist1, dist2 = self.cd(pcs1, pcs2)  # each (B, N) and (B, M)
        dist1 = torch.mean(torch.sqrt(dist1), dim=1)  # (B,)
        dist2 = torch.mean(torch.sqrt(dist2), dim=1)  # (B,)
        return torch.mean(dist1 + dist2) / 2.0  # scalar