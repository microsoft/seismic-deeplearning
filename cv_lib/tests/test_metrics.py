# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import numpy as np
from pytest import approx

from ignite.metrics import ConfusionMatrix, MetricsLambda

from cv_lib.segmentation.metrics import class_accuracy, mean_class_accuracy


# source repo:
# https://github.com/pytorch/ignite/blob/master/tests/ignite/metrics/test_confusion_matrix.py
def _get_y_true_y_pred():
    # Generate an image with labels 0 (background), 1, 2
    # 3 classes:
    y_true = np.zeros((30, 30), dtype=np.int)
    y_true[1:11, 1:11] = 1
    y_true[15:25, 15:25] = 2

    y_pred = np.zeros((30, 30), dtype=np.int)
    y_pred[20:30, 1:11] = 1
    y_pred[20:30, 20:30] = 2
    return y_true, y_pred


# source repo:
# https://github.com/pytorch/ignite/blob/master/tests/ignite/metrics/test_confusion_matrix.py
def _compute_th_y_true_y_logits(y_true, y_pred):
    # Create torch.tensor from numpy
    th_y_true = torch.from_numpy(y_true).unsqueeze(0)
    # Create logits torch.tensor:
    num_classes = max(np.max(y_true), np.max(y_pred)) + 1
    y_probas = np.ones((num_classes,) + y_true.shape) * -10
    for i in range(num_classes):
        y_probas[i, (y_pred == i)] = 720
    th_y_logits = torch.from_numpy(y_probas).unsqueeze(0)
    return th_y_true, th_y_logits


# Dependency metrics do not get updated automatically, so need to retrieve and
# update confusion matrix manually
def _get_cm(metriclambda):
    metrics = list(metriclambda.args)
    while metrics:
        metric = metrics[0]
        if isinstance(metric, ConfusionMatrix):
            return metric
        elif isinstance(metric, MetricsLambda):
            metrics.extend(metric.args)
        del metrics[0]


def test_class_accuracy():
    y_true, y_pred = _get_y_true_y_pred()

    ## Perfect prediction
    th_y_true, th_y_logits = _compute_th_y_true_y_logits(y_true, y_true)
    # Update metric
    output = (th_y_logits, th_y_true)
    acc_metric = class_accuracy(num_classes=3)
    acc_metric.update(output)

    # Retrieve and update confusion matrix
    metric_cm = _get_cm(acc_metric)
    # assert confusion matrix exists and is all zeroes
    assert metric_cm is not None
    assert torch.min(metric_cm.confusion_matrix) == 0.0 and torch.max(metric_cm.confusion_matrix) == 0.0
    metric_cm.update(output)

    # Expected result
    true_res = [1.0, 1.0, 1.0]
    res = acc_metric.compute().numpy()
    assert np.all(res == true_res), "Result {} vs. expected values {}".format(res, true_res)

    ## Imperfect prediction
    th_y_true, th_y_logits = _compute_th_y_true_y_logits(y_true, y_pred)
    # Update metric
    output = (th_y_logits, th_y_true)
    acc_metric = class_accuracy(num_classes=3)
    acc_metric.update(output)

    # Retrieve and update confusion matrix
    metric_cm = _get_cm(acc_metric)
    assert metric_cm is not None
    assert torch.min(metric_cm.confusion_matrix) == 0.0 and torch.max(metric_cm.confusion_matrix) == 0.0
    metric_cm.update(output)

    # Expected result
    true_res = [0.75, 0.0, 0.25]
    res = acc_metric.compute().numpy()
    assert np.all(res == true_res), "Result {} vs. expected values {}".format(res, true_res)


def test_mean_class_accuracy():
    y_true, y_pred = _get_y_true_y_pred()

    ## Perfect prediction
    th_y_true, th_y_logits = _compute_th_y_true_y_logits(y_true, y_true)
    # Update metric
    output = (th_y_logits, th_y_true)
    acc_metric = mean_class_accuracy(num_classes=3)
    acc_metric.update(output)

    # Retrieve and update confusion matrix
    metric_cm = _get_cm(acc_metric)
    metric_cm.update(output)

    # Expected result
    true_res = 1.0
    res = acc_metric.compute().numpy()
    assert res == approx(true_res), "Result {} vs. expected value {}".format(res, true_res)

    ## Imperfect prediction
    th_y_true, th_y_logits = _compute_th_y_true_y_logits(y_true, y_pred)
    # Update metric
    output = (th_y_logits, th_y_true)
    acc_metric = mean_class_accuracy(num_classes=3)
    acc_metric.update(output)

    # Retrieve and update confusion matrix
    metric_cm = _get_cm(acc_metric)
    metric_cm.update(output)

    # Expected result
    true_res = 1 / 3
    res = acc_metric.compute().numpy()
    assert res == approx(true_res), "Result {} vs. expected value {}".format(res, true_res)
