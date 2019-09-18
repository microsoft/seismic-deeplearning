import warnings

import numpy as np
import torch
from ignite.metrics import Metric


def do_kaggle_metric(predict, truth, threshold=0.5):
    N = len(predict)
    predict = predict.reshape(N, -1)
    truth = truth.reshape(N, -1)

    predict = predict > threshold
    truth = truth > 0.5
    intersection = truth & predict
    union = truth | predict
    iou = intersection.sum(1) / (union.sum(1) + 1e-8)

    # -------------------------------------------
    result = []
    precision = []
    is_empty_truth = truth.sum(1) == 0
    is_empty_predict = predict.sum(1) == 0

    threshold = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    for t in threshold:
        p = iou >= t

        tp = (~is_empty_truth) & (~is_empty_predict) & (iou > t)
        fp = (~is_empty_truth) & (~is_empty_predict) & (iou <= t)
        fn = (~is_empty_truth) & (is_empty_predict)
        fp_empty = (is_empty_truth) & (~is_empty_predict)
        tn_empty = (is_empty_truth) & (is_empty_predict)

        p = (tp + tn_empty) / (tp + tn_empty + fp + fp_empty + fn)

        result.append(np.column_stack((tp, fp, fn, tn_empty, fp_empty)))
        precision.append(p)

    result = np.array(result).transpose(1, 2, 0)
    precision = np.column_stack(precision)
    precision = precision.mean(1)

    return precision, result, threshold


class KaggleMetric(Metric):
    def __init__(self, output_transform=lambda x: x):
        super(KaggleMetric, self).__init__(output_transform=output_transform)

    def reset(self):
        self._predictions = torch.tensor([], dtype=torch.float32)
        self._targets = torch.tensor([], dtype=torch.long)

    def update(self, output):
        y_pred, y = output

        y_pred = y_pred.type_as(self._predictions)
        y = y.type_as(self._targets)

        self._predictions = torch.cat([self._predictions, y_pred], dim=0)
        self._targets = torch.cat([self._targets, y], dim=0)

        # Check once the signature and execution of compute_fn
        if self._predictions.shape == y_pred.shape:
            try:
                self.compute()
            except Exception as e:
                warnings.warn(
                    "Probably, there can be a problem with `compute_fn`:\n {}.".format(
                        e
                    ),
                    RuntimeWarning,
                )

    def compute(self):
        precision, _, _ = do_kaggle_metric(
            self._predictions.numpy(), self._targets.numpy(), 0.5
        )
        precision = precision.mean()
        return precision


def var_to_np(var):
    """Take a pytorch variable and make numpy
    """
    if type(var) in [np.array, np.ndarray]:
        return var

    # If input is list we do this for all elements
    if type(var) == type([]):
        out = []
        for v in var:
            out.append(var_to_np(v))
        return out

    # TODO: Replace this is from the original implementation and is a really bad idea
    try:
        var = var.cpu()
    except:
        None
    try:
        var = var.data
    except:
        None
    try:
        var = var.numpy()
    except:
        None

    if type(var) == tuple:
        var = var[0]
    return var


def pixel_wise_accuracy(predicted_class, labels):
    labels = var_to_np(labels)
    predicted_class = var_to_np(predicted_class)

    accuracies = {}
    for cls in np.unique(labels):
        if cls >= 0:
            accuracies["accuracy_class_" + str(cls)] = int(
                np.mean(predicted_class[labels == cls] == cls) * 100
            )
    accuracies["average_class_accuracy"] = np.mean([acc for acc in accuracies.values()])
    return accuracies

EPS = 1e-10



def overall_pixel_accuracy(hist):
    """Computes the total pixel accuracy.
    The overall pixel accuracy provides an intuitive
    approximation for the qualitative perception of the
    label when it is viewed in its overall shape but not
    its details.
    Args:
        hist: confusion matrix.
    Returns:
        overall_acc: the overall pixel accuracy.
    """
    correct = torch.diag(hist).sum()
    total = hist.sum()
    overall_acc = correct / (total + EPS)
    return overall_acc

def pixel_wise_accuracy2(predicted_class, labels, num_classes=1):
    hist = torch.zeros(num_classes+1).cuda()
    for i in range(num_classes+1):
        hist[i]=torch.mean((predicted_class[labels == i] == i).float()) * 100
    return torch.mean(hist)

class PixelwiseAccuracy(Metric):
    def __init__(self, output_transform=lambda x: x):
        super(PixelwiseAccuracy, self).__init__(output_transform=output_transform)
        self._threshold=0.5

    def reset(self):
        self._accuracies = []
        self._weights = []

    def update(self, output):
        y_pred, y = output
        y_pred[y_pred>self._threshold]=1
        y_pred[y_pred<=self._threshold]=0
        try:
            acc_dict = pixel_wise_accuracy(y_pred, y)
            print(acc_dict)
            self._accuracies.append(acc_dict["average_class_accuracy"])
            self._weights.append(y_pred.shape[0])  # Weight by batch size
        except Exception as e:
            warnings.warn(
                "Error computing accuracy:\n {}.".format(e), RuntimeWarning
            )

    def compute(self):
        return np.average(self._accuracies, weights=self._weights)
