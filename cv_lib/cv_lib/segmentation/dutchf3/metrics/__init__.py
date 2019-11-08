import numpy as np
import warnings

import numpy as np
import torch
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


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



class ConfusionMatrix(Metric):
    """Calculates confusion matrix for multi-class data.
    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must contain logits and has the following shape (batch_size, num_categories, ...)
    - `y` should have the following shape (batch_size, ...) and contains ground-truth class indices
        with or without the background class. During the computation, argmax of `y_pred` is taken to determine
        predicted classes.
    Args:
        num_classes (int): number of classes. See notes for more details.
        average (str, optional): confusion matrix values averaging schema: None, "samples", "recall", "precision".
            Default is None. If `average="samples"` then confusion matrix values are normalized by the number of seen
            samples. If `average="recall"` then confusion matrix values are normalized such that diagonal values
            represent class recalls. If `average="precision"` then confusion matrix values are normalized such that
            diagonal values represent class precisions.
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        device (str of torch.device, optional): device specification in case of distributed computation usage.
            In most of the cases, it can be defined as "cuda:local_rank" or "cuda"
            if already set `torch.cuda.set_device(local_rank)`. By default, if a distributed process group is
            initialized and available, device is set to `cuda`.
    Note:
        In case of the targets `y` in `(batch_size, ...)` format, target indices between 0 and `num_classes` only
        contribute to the confusion matrix and others are neglected. For example, if `num_classes=20` and target index
        equal 255 is encountered, then it is filtered out.
    """

    def __init__(self, num_classes, average=None, output_transform=lambda x: x, device=None):
        if average is not None and average not in ("samples", "recall", "precision"):
            raise ValueError("Argument average can None or one of ['samples', 'recall', 'precision']")

        self.num_classes = num_classes
        self._num_examples = 0
        self.average = average
        self.confusion_matrix = None
        self._device=device
        super(ConfusionMatrix, self).__init__(output_transform=output_transform)

    def reset(self):
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes,
                                            dtype=torch.int64,
                                            device=self._device)
        self._num_examples = 0

    def _check_shape(self, output):
        y_pred, y = output

        if y_pred.ndimension() < 2:
            raise ValueError("y_pred must have shape (batch_size, num_categories, ...), "
                             "but given {}".format(y_pred.shape))

        if y_pred.shape[1] != self.num_classes:
            raise ValueError("y_pred does not have correct number of categories: {} vs {}"
                             .format(y_pred.shape[1], self.num_classes))

        if not (y.ndimension() + 1 == y_pred.ndimension()):
            raise ValueError("y_pred must have shape (batch_size, num_categories, ...) and y must have "
                             "shape of (batch_size, ...), "
                             "but given {} vs {}.".format(y.shape, y_pred.shape))

        y_shape = y.shape
        y_pred_shape = y_pred.shape

        if y.ndimension() + 1 == y_pred.ndimension():
            y_pred_shape = (y_pred_shape[0],) + y_pred_shape[2:]

        if y_shape != y_pred_shape:
            raise ValueError("y and y_pred must have compatible shapes.")

    def update(self, output):
        self._check_shape(output)
        y_pred, y = output

        self._num_examples += y_pred.shape[0]

        # target is (batch_size, ...)
        y_pred = torch.argmax(y_pred, dim=1).flatten()
        y = y.flatten()

        target_mask = (y >= 0) & (y < self.num_classes)
        y = y[target_mask]
        y_pred = y_pred[target_mask]

        indices = self.num_classes * y + y_pred
        m = torch.bincount(indices, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += m.to(self.confusion_matrix)

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Confusion matrix must have at least one example before it can be computed.')
        if self.average:
            self.confusion_matrix = self.confusion_matrix.float()
            if self.average == "samples":
                return self.confusion_matrix / self._num_examples
            elif self.average == "recall":
                return self.confusion_matrix / (self.confusion_matrix.sum(dim=1) + 1e-15)
            elif self.average == "precision":
                return self.confusion_matrix / (self.confusion_matrix.sum(dim=0) + 1e-15)
        return self.confusion_matrix


class MeanIoU(ConfusionMatrix):
    def compute(self):
        hist = self.confusion_matrix.cpu().numpy()
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
        mean_iu = np.nanmean(iu)
        return mean_iu


class PixelwiseAccuracy(ConfusionMatrix):
    def compute(self):
        hist = self.confusion_matrix.cpu().numpy()
        acc = np.diag(hist).sum() / hist.sum()
        return acc

class FrequencyWeightedIoU(ConfusionMatrix):
    def compute(self):
        hist = self.confusion_matrix.cpu().numpy()
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
        freq = hist.sum(axis=1) / hist.sum() # fraction of the pixels that come from each class
        fwiou = (freq[freq > 0] * iu[freq > 0]).sum()
        return fwiou


class MeanClassAccuracy(ConfusionMatrix):
    def compute(self):
        hist = self.confusion_matrix.cpu().numpy()
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        mean_acc_cls = np.nanmean(acc_cls)
        return mean_acc_cls