# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import torch
import torch.distributed as dist

# from cv_lib.segmentation.dutchf3 import metrics
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric


def _torch_hist(label_true, label_pred, n_class):
    """Calculates the confusion matrix for the labels
    
    Args:
        label_true ([type]): [description]
        label_pred ([type]): [description]
        n_class ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    # TODO Add exceptions
    assert len(label_true.shape) == 1, "Labels need to be 1D"
    assert len(label_pred.shape) == 1, "Predictions need to be 1D"
    mask = (label_true >= 0) & (label_true < n_class)
    hist = torch.bincount(
        n_class * label_true[mask] + label_pred[mask], minlength=n_class ** 2
    ).reshape(n_class, n_class)
    return hist


class OldConfusionMatrix(Metric):
    def __init__(self, num_classes, device, output_transform=lambda x: x):
        self._num_classes = num_classes
        self._device = device
        super(OldConfusionMatrix, self).__init__(output_transform=output_transform)

    def reset(self):
        self._confusion_matrix = torch.zeros(
            (self._num_classes, self._num_classes), dtype=torch.long
        ).to(self._device)

    def update(self, output):
        y_pred, y = output
        # TODO: Make assertion exception
        assert y.shape == y_pred.max(1)[1].squeeze().shape, "Shape not the same"
        self._confusion_matrix += _torch_hist(
            torch.flatten(y),
            torch.flatten(y_pred.max(1)[1].squeeze()),  # Get the maximum index
            self._num_classes,
        )

    def compute(self):
        return self._confusion_matrix.cpu().numpy()


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


class AverageMetric(Metric):
    def __init__(self, world_size, batch_size, output_transform=lambda x: x):
        super(AverageMetric, self).__init__(output_transform=output_transform)
        self._world_size = world_size
        self._batch_size = batch_size
        self._metric_name = "Metric"

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    @torch.no_grad()
    def update(self, output):
        reduced_metric = reduce_tensor(output, self._world_size)
        self._sum += reduced_metric * self._batch_size
        self._num_examples += self._batch_size

    @torch.no_grad()
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                f"{self._metric_name} must have at least one example before it can be \
                     computed."
            )
        return self._sum / self._num_examples


class LossMetric(AverageMetric):
    def __init__(self, loss_fn, world_size, batch_size, output_transform=lambda x: x):
        super(LossMetric, self).__init__(
            world_size, batch_size, output_transform=output_transform
        )
        self._loss_fn = loss_fn
        self._metric_name = "Loss"

    def update(self, output):
        pred, y = output
        loss = self._loss_fn(pred, y)
        super().update(loss)


class ConfusionMatrix(OldConfusionMatrix):
    def compute(self):
        reduced_metric = sum_reduce_tensor(self._confusion_matrix)
        return reduced_metric.cpu().numpy()


class PixelwiseAccuracy(ConfusionMatrix):
    def compute(self):
        hist = super(PixelwiseAccuracy, self).compute()
        acc = np.diag(hist).sum() / hist.sum()
        return acc


class MeanIoU(ConfusionMatrix):
    def compute(self):
        hist = super(MeanIoU, self).compute()
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        return mean_iu


class FrequencyWeightedIoU(ConfusionMatrix):
    def compute(self):
        hist = super(FrequencyWeightedIoU, self).compute()
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        freq = (
            hist.sum(axis=1) / hist.sum()
        )  # fraction of the pixels that come from each class
        fwiou = (freq[freq > 0] * iu[freq > 0]).sum()
        return fwiou


class MeanClassAccuracy(ConfusionMatrix):
    def compute(self):
        hist = super(MeanClassAccuracy, self).compute()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        mean_acc_cls = np.nanmean(acc_cls)
        return mean_acc_cls
