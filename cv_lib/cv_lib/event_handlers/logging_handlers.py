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


@curry
def log_metrics(log_msg, engine, metrics_dict={"pixacc": "Avg accuracy :", "nll": "Avg loss :"}):
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


# TODO: remove Evaluator once other train.py scripts are updated
class Evaluator:
    def __init__(self, evaluation_engine, data_loader):
        self._evaluation_engine = evaluation_engine
        self._data_loader = data_loader

    def __call__(self, engine):
        self._evaluation_engine.run(self._data_loader)
