# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from ignite.engine.engine import Engine
from toolz import curry
from torch.nn import functional as F


def _upscale_model_output(y_pred, y):
    ph, pw = y_pred.size(2), y_pred.size(3)
    h, w = y.size(2), y.size(3)
    if ph != h or pw != w:
        y_pred = F.upsample(input=y_pred, size=(h, w), mode="bilinear")
    return y_pred


def create_supervised_trainer(
    model,
    optimizer,
    loss_fn,
    prepare_batch,
    device=None,
    non_blocking=False,
    output_transform=lambda x, y, y_pred, loss: {"loss": loss.item()},
):
    """Factory function for creating a trainer for supervised segmentation models.

    Args:
        model (`torch.nn.Module`): the model to train.
        optimizer (`torch.optim.Optimizer`): the optimizer to use.
        loss_fn (torch.nn loss function): the loss function to use.
        prepare_batch (callable): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y, patch_id, patch_locations)`.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.

    Note: `engine.state.output` for this engine is defined by `output_transform` parameter and is the loss
        of the processed batch by default.

    Returns:
        Engine: a trainer engine with supervised update function.
    """
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y, ids, patch_locations = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        y_pred = _upscale_model_output(y_pred, y)
        loss = loss_fn(y_pred.squeeze(1), y.squeeze(1))
        loss.backward()
        optimizer.step()
        return output_transform(x, y, y_pred, loss)

    return Engine(_update)


@curry
def val_transform(x, y, y_pred, ids, patch_locations):
    return {
        "image": x,
        "y_pred": y_pred.detach(),
        "mask": y.detach(),
        "ids": ids,
        "patch_locations": patch_locations,
    }


def create_supervised_evaluator(
    model, prepare_batch, metrics=None, device=None, non_blocking=False, output_transform=val_transform,
):
    """Factory function for creating an evaluator for supervised segmentation models.

    Args:
       model (`torch.nn.Module`): the model to train.
       prepare_batch (callable): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y, patch_id, patch_locations)`.
       metrics (dict of str - :class:`~ignite.metrics.Metric`): a map of metric names to Metrics.
       device (str, optional): device type specification (default: None).
           Applies to both model and batches.
       non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
           with respect to the host. For other cases, this argument has no effect.
       output_transform (callable, optional): function that receives 'x', 'y', 'y_pred' and returns value
           to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits
           output expected by metrics. If you change it you should use `output_transform` in metrics.

    Note: `engine.state.output` for this engine is defind by `output_transform` parameter and is
       a tuple of `(batch_pred, batch_y)` by default.

    Returns:
       Engine: an evaluator engine with supervised inference function.
    """
    metrics = metrics or {}

    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y, ids, patch_locations = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x)
            y_pred = _upscale_model_output(y_pred, x)
            return output_transform(x, y, y_pred, ids, patch_locations)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
