import torch.distributed as dist
from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError
import torch
from cv_lib.segmentation.dutchf3 import metrics
import numpy as np

@torch.no_grad()
def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt

@torch.no_grad()
def sum_reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    return rt

@torch.no_grad()
def gather_tensor(tensor, world_size):
    gather_t = [torch.ones_like(tensor).cuda() for _ in range(dist.get_world_size())]
    dist.all_gather(gather_t, tensor)
    return gather_t


class ConfusionMatrix(metrics.ConfusionMatrix):
    def compute(self):
        reduced_metric = sum_reduce_tensor(self._confusion_matrix)
        return reduced_metric.cpu().numpy()

class PixelwiseAccuracy(ConfusionMatrix):
    def compute(self):
        hist = super(PixelwiseAccuracy,self).compute()
        acc = np.diag(hist).sum() / hist.sum()
        return acc

class MeanIoU(ConfusionMatrix):
    def compute(self):
        hist = super(MeanIoU, self).compute()
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
        mean_iu = np.nanmean(iu)
        return mean_iu


class FrequencyWeightedIoU(ConfusionMatrix):
    def compute(self):
        hist = super(FrequencyWeightedIoU, self).compute()
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
        freq = hist.sum(axis=1) / hist.sum() # fraction of the pixels that come from each class
        fwiou = (freq[freq > 0] * iu[freq > 0]).sum()
        return fwiou


class MeanClassAccuracy(ConfusionMatrix):
    def compute(self):
        hist = super(MeanClassAccuracy, self).compute()
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        mean_acc_cls = np.nanmean(acc_cls)
        return mean_acc_cls