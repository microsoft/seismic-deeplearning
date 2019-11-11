# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import ignite


def pixelwise_accuracy(num_classes, output_transform=lambda x: x, device=None):
    """Calculates class accuracy

    Args:
        num_classes (int): number of classes
        output_transform (callable, optional): a callable that is used to transform the
            output into the form expected by the metric.

    Returns:
        MetricsLambda

    """
    cm = ignite.metrics.ConfusionMatrix(num_classes=num_classes, output_transform=output_transform, device=device)
    # Increase floating point precision and pass to CPU
    cm = cm.type(torch.DoubleTensor)

    pix_cls = ignite.metrics.confusion_matrix.cmAccuracy(cm)

    return pix_cls


def class_accuracy(num_classes, output_transform=lambda x: x, device=None):
    """Calculates class accuracy

    Args:
        num_classes (int): number of classes
        output_transform (callable, optional): a callable that is used to transform the
            output into the form expected by the metric.

    Returns:
        MetricsLambda

    """
    cm = ignite.metrics.ConfusionMatrix(num_classes=num_classes, output_transform=output_transform, device=device)
    # Increase floating point precision and pass to CPU
    cm = cm.type(torch.DoubleTensor)

    acc_cls = cm.diag() / (cm.sum(dim=1) + 1e-15)

    return acc_cls


def mean_class_accuracy(num_classes, output_transform=lambda x: x, device=None):
    """Calculates mean class accuracy

    Args:
        num_classes (int): number of classes
        output_transform (callable, optional): a callable that is used to transform the
            output into the form expected by the metric.

    Returns:
        MetricsLambda

    """
    return class_accuracy(num_classes=num_classes, output_transform=output_transform, device=device).mean()


def class_iou(num_classes, output_transform=lambda x: x, device=None, ignore_index=None):
    """Calculates per-class intersection-over-union

    Args:
        num_classes (int): number of classes
        output_transform (callable, optional): a callable that is used to transform the
            output into the form expected by the metric.

    Returns:
        MetricsLambda

    """
    cm = ignite.metrics.ConfusionMatrix(num_classes=num_classes, output_transform=output_transform, device=device)
    return ignite.metrics.IoU(cm, ignore_index=ignore_index)


def mean_iou(num_classes, output_transform=lambda x: x, device=None, ignore_index=None):
    """Calculates mean intersection-over-union

    Args:
        num_classes (int): number of classes
        output_transform (callable, optional): a callable that is used to transform the
            output into the form expected by the metric.

    Returns:
        MetricsLambda

    """
    cm = ignite.metrics.ConfusionMatrix(num_classes=num_classes, output_transform=output_transform, device=device)
    return ignite.metrics.mIoU(cm, ignore_index=ignore_index)
