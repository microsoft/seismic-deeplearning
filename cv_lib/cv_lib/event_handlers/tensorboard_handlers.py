# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from toolz import curry
import torchvision
import logging
import logging.config

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")


def create_summary_writer(log_dir):
    writer = SummaryWriter(logdir=log_dir)
    return writer


def _log_model_output(log_label, summary_writer, engine):
    summary_writer.add_scalar(log_label, engine.state.output["loss"], engine.state.iteration)


@curry
def log_training_output(summary_writer, engine):
    _log_model_output("training/loss", summary_writer, engine)


@curry
def log_validation_output(summary_writer, engine):
    _log_model_output("validation/loss", summary_writer, engine)


@curry
def log_lr(summary_writer, optimizer, log_interval, engine):
    """[summary]
    
    Args:
        optimizer ([type]): [description]
        log_interval ([type]): iteration or epoch
        summary_writer ([type]): [description]
        engine ([type]): [description]
    """
    lr = [param_group["lr"] for param_group in optimizer.param_groups]
    summary_writer.add_scalar("lr", lr[0], getattr(engine.state, log_interval))


_DEFAULT_METRICS = {"accuracy": "Avg accuracy :", "nll": "Avg loss :"}


@curry
def log_metrics(summary_writer, train_engine, log_interval, engine, metrics_dict=_DEFAULT_METRICS):
    metrics = engine.state.metrics
    for m in metrics_dict:
        summary_writer.add_scalar(metrics_dict[m], metrics[m], getattr(train_engine.state, log_interval))


def create_image_writer(summary_writer, label, output_variable, normalize=False, transform_func=lambda x: x):
    logger = logging.getLogger(__name__)

    def write_to(engine):
        try:
            data_tensor = transform_func(engine.state.output[output_variable])
            image_grid = torchvision.utils.make_grid(data_tensor, normalize=normalize, scale_each=True)
            summary_writer.add_image(label, image_grid, engine.state.epoch)
        except KeyError:
            logger.warning("Predictions and or ground truth labels not available to report")

    return write_to
