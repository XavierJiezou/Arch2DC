from models.pcn import PCN
from models.adapointr import AdaPoinTr
from models.foldingnet import FoldingNet
from models.adapointr_nomask import ToothNoMaskAdaPoinTr
from models.pointattn import PointAttN
from models.pointattn_nomask import ToothNoMaskPointAttN
from models.odgnet_nomask import ToothNoMaskODGNet

# Backward-compatible alias for older scripts/checkpoint helpers.
ToothMaskAdaPoinTr = ToothNoMaskAdaPoinTr
