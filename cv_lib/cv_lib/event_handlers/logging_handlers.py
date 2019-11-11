# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import logging.config
from toolz import curry

import numpy as np

np.set_printoptions(precision=3)


@curry
def log_training_output(engine, log_interval=100):
    logger = logging.getLogger(__name__)

    if engine.state.iteration % log_interval == 0:
        logger.info(f"Epoch: {engine.state.epoch} Iter: {engine.state.iteration} loss {engine.state.output['loss']}")


@curry
def log_lr(optimizer, engine):
    logger = logging.getLogger(__name__)
    lr = [param_group["lr"] for param_group in optimizer.param_groups]
    logger.info(f"lr - {lr}")


_DEFAULT_METRICS = {"pixacc": "Avg accuracy :", "nll": "Avg loss :"}


@curry
def log_metrics(log_msg, engine, metrics_dict=_DEFAULT_METRICS):
    logger = logging.getLogger(__name__)
    metrics = engine.state.metrics
    metrics_msg = " ".join([f"{metrics_dict[k]} {metrics[k]:.2f}" for k in metrics_dict])
    logger.info(f"{log_msg} - Epoch {engine.state.epoch} [{engine.state.max_epochs}] " + metrics_msg)


@curry
def log_class_metrics(log_msg, engine, metrics_dict):
    logger = logging.getLogger(__name__)
    metrics = engine.state.metrics
    metrics_msg = "\n".join(f"{metrics_dict[k]} {metrics[k].numpy()}" for k in metrics_dict)
    logger.info(f"{log_msg} - Epoch {engine.state.epoch} [{engine.state.max_epochs}]\n" + metrics_msg)


class Evaluator:
    def __init__(self, evaluation_engine, data_loader):
        self._evaluation_engine = evaluation_engine
        self._data_loader = data_loader

    def __call__(self, engine):
        self._evaluation_engine.run(self._data_loader)


class HorovodLRScheduler:
    """
    Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
    accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
    the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
    """

    def __init__(
        self, base_lr, warmup_epochs, cluster_size, data_loader, optimizer, batches_per_allreduce,
    ):
        self._warmup_epochs = warmup_epochs
        self._cluster_size = cluster_size
        self._data_loader = data_loader
        self._optimizer = optimizer
        self._base_lr = base_lr
        self._batches_per_allreduce = batches_per_allreduce
        self._logger = logging.getLogger(__name__)

    def __call__(self, engine):
        epoch = engine.state.epoch
        if epoch < self._warmup_epochs:
            epoch += float(engine.state.iteration + 1) / len(self._data_loader)
            lr_adj = 1.0 / self._cluster_size * (epoch * (self._cluster_size - 1) / self._warmup_epochs + 1)
        elif epoch < 30:
            lr_adj = 1.0
        elif epoch < 60:
            lr_adj = 1e-1
        elif epoch < 80:
            lr_adj = 1e-2
        else:
            lr_adj = 1e-3
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = self._base_lr * self._cluster_size * self._batches_per_allreduce * lr_adj
            self._logger.debug(f"Adjust learning rate {param_group['lr']}")
