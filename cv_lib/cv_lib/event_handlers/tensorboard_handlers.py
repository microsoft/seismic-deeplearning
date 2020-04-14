# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torchvision
from toolz import curry

from cv_lib.segmentation.dutchf3.utils import np_to_tb
from deepseismic_interpretation.dutchf3.data import decode_segmap

from tensorboardX import SummaryWriter


def create_summary_writer(log_dir):
    writer = SummaryWriter(logdir=log_dir)
    return writer


def _transform_image(output_tensor):
    output_tensor = output_tensor.cpu()
    return torchvision.utils.make_grid(output_tensor, normalize=True, scale_each=True)


def _transform_pred(output_tensor, n_classes):
    output_tensor = output_tensor.squeeze().cpu().numpy()
    decoded = decode_segmap(output_tensor, n_classes)  # 16x3x256x256
    return torchvision.utils.make_grid(np_to_tb(decoded), normalize=False, scale_each=False)


def _log_model_output(log_label, summary_writer, engine):
    summary_writer.add_scalar(log_label, engine.state.output["loss"], engine.state.iteration)


@curry
def log_training_output(summary_writer, engine):
    _log_model_output("Training/loss", summary_writer, engine)


@curry
def log_validation_output(summary_writer, engine):
    _log_model_output("Validation/loss", summary_writer, engine)


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


def log_results(engine, evaluator, summary_writer, n_classes, stage):
    epoch = engine.state.epoch  # note: the evaluator epoch is always 1.. useless
    metrics = evaluator.state.metrics
    outputs = evaluator.state.output

    # Log Metrics:
    summary_writer.add_scalar(f"{stage}/mIoU", metrics["mIoU"], epoch)
    summary_writer.add_scalar(f"{stage}/nll", metrics["nll"], epoch)
    summary_writer.add_scalar(f"{stage}/mca", metrics["mca"], epoch)
    summary_writer.add_scalar(f"{stage}/pixacc", metrics["pixacc"], epoch)


@curry
def log_metrics(
    summary_writer, train_engine, log_interval, engine, metrics_dict={"pixacc": "Avg accuracy :", "nll": "Avg loss :"}
):
    metrics = engine.state.metrics
    for m in metrics_dict:
        summary_writer.add_scalar(metrics_dict[m], metrics[m], getattr(train_engine.state, log_interval))

    # Log Images:
    image = outputs["image"]
    mask = outputs["mask"]
    y_pred = outputs["y_pred"].max(1, keepdim=True)[1]
    VISUALIZATION_LIMIT = 16


def create_image_writer(summary_writer, label, output_variable, normalize=False, transform_func=lambda x: x):
    logger = logging.getLogger(__name__)

    def write_to(engine):
        try:
            data_tensor = transform_func(engine.state.output[output_variable])
            image_grid = torchvision.utils.make_grid(data_tensor, normalize=normalize, scale_each=True)
            summary_writer.add_image(label, image_grid, engine.state.epoch)
        except KeyError:
            logger.warning("Predictions and or ground truth labels not available to report")

    summary_writer.add_image(f"{stage}/Image", _transform_image(image), epoch)
    summary_writer.add_image(f"{stage}/Mask", _transform_pred(mask, n_classes), epoch)
    summary_writer.add_image(f"{stage}/Pred", _transform_pred(y_pred, n_classes), epoch)
