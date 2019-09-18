import numpy as np
import warnings

import numpy as np
import torch
from ignite.metrics import Metric


# class runningScore(object):
#     def __init__(self, n_classes):
#         self.n_classes = n_classes
#         self.confusion_matrix = np.zeros((n_classes, n_classes))

#     def _fast_hist(self, label_true, label_pred, n_class):
#         mask = (label_true >= 0) & (label_true < n_class)
#         hist = np.bincount(
#             n_class * label_true[mask].astype(int) + label_pred[mask],
#             minlength=n_class ** 2,
#         ).reshape(n_class, n_class)
#         return hist

#     def update(self, label_trues, label_preds):
#         for lt, lp in zip(label_trues, label_preds):
#             self.confusion_matrix += self._fast_hist(
#                 lt.flatten(), lp.flatten(), self.n_classes
#             )

#     def get_scores(self):
#         """Returns accuracy score evaluation result.
#             - overall accuracy
#             - mean accuracy
#             - mean IU
#             - fwavacc
#         """
#         hist = self.confusion_matrix
#         acc = np.diag(hist).sum() / hist.sum()
#         acc_cls = np.diag(hist) / hist.sum(axis=1)
#         mean_acc_cls = np.nanmean(acc_cls)
#         iu = np.diag(hist) / (
#             hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
#         )
#         mean_iu = np.nanmean(iu)
#         freq = (
#             hist.sum(axis=1) / hist.sum()
#         )  # fraction of the pixels that come from each class
#         fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
#         cls_iu = dict(zip(range(self.n_classes), iu))

#         return (
#             {
#                 "Pixel Acc: ": acc,
#                 "Class Accuracy: ": acc_cls,
#                 "Mean Class Acc: ": mean_acc_cls,
#                 "Freq Weighted IoU: ": fwavacc,
#                 "Mean IoU: ": mean_iu,
#                 "confusion_matrix": self.confusion_matrix,
#             },
#             cls_iu,
#         )

#     def reset(self):
#         self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


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
    def __init__(self, num_classes, device, output_transform=lambda x: x):
        self._num_classes = num_classes
        self._device = device
        super(ConfusionMatrix, self).__init__(output_transform=output_transform)

    def reset(self):
        self._confusion_matrix = torch.zeros((self._num_classes, self._num_classes),dtype=torch.long).to(self._device)

    def update(self, output):
        y_pred, y = output
        #TODO: Make assertion exception
        assert y.shape==y_pred.max(1)[1].squeeze().shape, "Shape not the same"
        self._confusion_matrix += _torch_hist(
            torch.flatten(y),
            torch.flatten(y_pred.max(1)[1].squeeze()), # Get the maximum index
            self._num_classes,
        )

    def compute(self):
        return self._confusion_matrix.cpu().numpy()


class MeanIoU(ConfusionMatrix):
    def compute(self):
        hist = self._confusion_matrix.cpu().numpy()
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
        mean_iu = np.nanmean(iu)
        return mean_iu


class PixelwiseAccuracy(ConfusionMatrix):
    def compute(self):
        hist = self._confusion_matrix.cpu().numpy()
        acc = np.diag(hist).sum() / hist.sum()
        return acc

class FrequencyWeightedIoU(ConfusionMatrix):
    def compute(self):
        hist = self._confusion_matrix.cpu().numpy()
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
        freq = hist.sum(axis=1) / hist.sum() # fraction of the pixels that come from each class
        fwiou = (freq[freq > 0] * iu[freq > 0]).sum()
        return fwiou


class MeanClassAccuracy(ConfusionMatrix):
    def compute(self):
        hist = self._confusion_matrix.cpu().numpy()
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        mean_acc_cls = np.nanmean(acc_cls)
        return mean_acc_cls