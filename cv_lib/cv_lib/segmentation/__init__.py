# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from toolz import curry
import torch.nn.functional as F


@curry
def extract_metric_from(metric, engine):
    metrics = engine.state.metrics
    return metrics[metric]


@curry
def padded_val_transform(pad_left, fine_size, x, y, y_pred):
    y_pred = y_pred[:, :, pad_left : pad_left + fine_size, pad_left : pad_left + fine_size].contiguous()
    return {"image": x, "y_pred": F.sigmoid(y_pred).detach(), "mask": y.detach()}
