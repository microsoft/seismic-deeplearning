# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from ignite.engine.engine import Engine, State, Events
from ignite.utils import convert_tensor
import torch.nn.functional as F
from toolz import curry
from torch.nn import functional as F
import numpy as np


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
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        y_pred = _upscale_model_output(y_pred, y)
        loss = loss_fn(y_pred.squeeze(1), y.squeeze(1))
        loss.backward()
        optimizer.step()
        return output_transform(x, y, y_pred, loss)

    return Engine(_update)


@curry
def val_transform(x, y, y_pred):
    return {"image": x, "y_pred": y_pred.detach(), "mask": y.detach()}


def create_supervised_evaluator(
    model,
    prepare_batch,
    metrics=None,
    device=None,
    non_blocking=False,
    output_transform=val_transform,
):
    metrics = metrics or {}

    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x)
            y_pred = _upscale_model_output(y_pred, x)
            return output_transform(x, y, y_pred)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_supervised_trainer_apex(
    model,
    optimizer,
    loss_fn,
    prepare_batch,
    device=None,
    non_blocking=False,
    output_transform=lambda x, y, y_pred, loss: {"loss": loss.item()},
):
    from apex import amp

    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        loss = loss_fn(y_pred.squeeze(1), y.squeeze(1))
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        return output_transform(x, y, y_pred, loss)

    return Engine(_update)


# def create_supervised_evaluator_apex(
#     model,
#     prepare_batch,
#     metrics=None,
#     device=None,
#     non_blocking=False,
#     output_transform=lambda x, y, y_pred: (x, y, pred),
# ):
#     metrics = metrics or {}

#     if device:
#         model.to(device)

#     def _inference(engine, batch):
#         model.eval()
#         with torch.no_grad():
#             x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
#             y_pred = model(x)
#             return output_transform(x, y, y_pred)

#     engine = Engine(_inference)

#     for name, metric in metrics.items():
#         metric.attach(engine, name)

#     return engine

