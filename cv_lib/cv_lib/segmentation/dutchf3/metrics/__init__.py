from toolz import curry
import torch
import ignite


def class_accuracy(num_classes, output_transform=lambda x: x):
    """Calculates class accuracy

    Args:
        num_classes (int): number of classes
        output_transform (callable, optional): a callable that is used to transform the
            output into the form expected by the metric.

    Returns:
        MetricsLambda

    """
    cm = ignite.metrics.ConfusionMatrix(
        num_classes=num_classes, output_transform=output_transform
    )
    # Increase floating point precision and pass to CPU
    cm = cm.type(torch.DoubleTensor)

    acc_cls = cm.diag() / (cm.sum(dim=1) + 1e-15)

    return acc_cls


def mean_class_accuracy(num_classes, output_transform=lambda x: x):
    """Calculates mean class accuracy

    Args:
        num_classes (int): number of classes
        output_transform (callable, optional): a callable that is used to transform the 
            output into the form expected by the metric.

    Returns:
        MetricsLambda

    """
    return class_accuracy(
        num_classes=num_classes, output_transform=output_transform
    ).mean()


def class_iou(num_classes, output_transform=lambda x: x):
    """Calculates per-class intersection-over-union

    Args:
        num_classes (int): number of classes
        output_transform (callable, optional): a callable that is used to transform the
            output into the form expected by the metric.

    Returns:
        MetricsLambda

    """
    cm = ignite.metrics.ConfusionMatrix(
        num_classes=num_classes, output_transform=output_transform
    )
    return ignite.metrics.IoU(cm)


def mean_iou(num_classes, output_transform=lambda x: x):
    """Calculates mean intersection-over-union

    Args:
        num_classes (int): number of classes
        output_transform (callable, optional): a callable that is used to transform the
            output into the form expected by the metric.

    Returns:
        MetricsLambda

    """
    cm = ignite.metrics.ConfusionMatrix(
        num_classes=num_classes, output_transform=output_transform
    )
    return ignite.metrics.mIoU(cm)


@curry
def extract_metric_from(metric, engine):
    """Return a specific metric from ignite engine"""
    metrics = engine.state.metrics
    return metrics[metric]
