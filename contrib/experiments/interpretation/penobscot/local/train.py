# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# To Test:
# python train.py TRAIN.END_EPOCH 1 TRAIN.SNAPSHOTS 1 --cfg "configs/seresnet_unet.yaml" --debug
#
# /* spell-checker: disable */
"""Train models on Penobscot dataset

Trains models using PyTorch
Uses a warmup schedule that then goes into a cyclic learning rate

Time to run on single V100 for 300 epochs: 3.5 days
"""

import logging
import logging.config
from os import path

import fire
import numpy as np
import torch
from albumentations import Compose, HorizontalFlip, Normalize, PadIfNeeded, Resize
from ignite.contrib.handlers import CosineAnnealingScheduler
from ignite.engine import Events
from ignite.metrics import Loss
from ignite.utils import convert_tensor
from toolz import compose
from torch.utils import data

from cv_lib.event_handlers import SnapshotHandler, logging_handlers, tensorboard_handlers
from cv_lib.event_handlers.logging_handlers import Evaluator
from cv_lib.event_handlers.tensorboard_handlers import create_image_writer, create_summary_writer
from cv_lib.segmentation import extract_metric_from, models
from cv_lib.segmentation.dutchf3.utils import current_datetime, generate_path, git_branch, git_hash, np_to_tb
from cv_lib.segmentation.metrics import class_accuracy, class_iou, mean_class_accuracy, mean_iou, pixelwise_accuracy
from cv_lib.segmentation.penobscot.engine import create_supervised_evaluator, create_supervised_trainer
from cv_lib.utils import load_log_configuration
from deepseismic_interpretation.dutchf3.data import decode_segmap
from deepseismic_interpretation.penobscot.data import get_patch_dataset
from default import _C as config
from default import update_config

mask_value = 255


def _prepare_batch(batch, device=None, non_blocking=False):
    x, y, ids, patch_locations = batch
    return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        convert_tensor(y, device=device, non_blocking=non_blocking),
        ids,
        patch_locations,
    )


def run(*options, cfg=None, debug=False):
    """Run training and validation of model

    Notes:
        Options can be passed in via the options argument and loaded from the cfg file
        Options loaded from default.py will be overridden by those loaded from cfg file
        Options passed in via options argument will override those loaded from cfg file
    
    Args:
        *options (str, int, optional): Options used to overide what is loaded from the
                                    config. To see what options are available consult
                                    default.py
        cfg (str, optional): Location of config file to load. Defaults to None.
        debug (bool): Places scripts in debug/test mode and only executes a few iterations
    """

    update_config(config, options=options, config_file=cfg)

    # we will write the model under outputs / config_file_name / model_dir
    config_file_name = "default_config" if not cfg else cfg.split("/")[-1].split(".")[0]

    # Start logging
    load_log_configuration(config.LOG_CONFIG)
    logger = logging.getLogger(__name__)
    logger.debug(config.WORKERS)
    epochs_per_cycle = config.TRAIN.END_EPOCH // config.TRAIN.SNAPSHOTS
    torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK

    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(seed=config.SEED)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    # Setup Augmentations
    basic_aug = Compose(
        [
            Normalize(mean=(config.TRAIN.MEAN,), std=(config.TRAIN.STD,), max_pixel_value=config.TRAIN.MAX,),
            PadIfNeeded(
                min_height=config.TRAIN.PATCH_SIZE,
                min_width=config.TRAIN.PATCH_SIZE,
                border_mode=config.OPENCV_BORDER_CONSTANT,
                always_apply=True,
                mask_value=mask_value,
                value=0,
            ),
            Resize(
                config.TRAIN.AUGMENTATIONS.RESIZE.HEIGHT, config.TRAIN.AUGMENTATIONS.RESIZE.WIDTH, always_apply=True,
            ),
            PadIfNeeded(
                min_height=config.TRAIN.AUGMENTATIONS.PAD.HEIGHT,
                min_width=config.TRAIN.AUGMENTATIONS.PAD.WIDTH,
                border_mode=config.OPENCV_BORDER_CONSTANT,
                always_apply=True,
                mask_value=mask_value,
                value=0,
            ),
        ]
    )
    if config.TRAIN.AUGMENTATION:
        train_aug = Compose([basic_aug, HorizontalFlip(p=0.5)])
        val_aug = basic_aug
    else:
        train_aug = val_aug = basic_aug

    PenobscotDataset = get_patch_dataset(config)

    train_set = PenobscotDataset(
        config.DATASET.ROOT,
        config.TRAIN.PATCH_SIZE,
        config.TRAIN.STRIDE,
        split="train",
        transforms=train_aug,
        n_channels=config.MODEL.IN_CHANNELS,
        complete_patches_only=config.TRAIN.COMPLETE_PATCHES_ONLY,
    )

    val_set = PenobscotDataset(
        config.DATASET.ROOT,
        config.TRAIN.PATCH_SIZE,
        config.TRAIN.STRIDE,
        split="val",
        transforms=val_aug,
        n_channels=config.MODEL.IN_CHANNELS,
        complete_patches_only=config.VALIDATION.COMPLETE_PATCHES_ONLY,
    )
    logger.info(train_set)
    logger.info(val_set)
    n_classes = train_set.n_classes

    train_loader = data.DataLoader(
        train_set, batch_size=config.TRAIN.BATCH_SIZE_PER_GPU, num_workers=config.WORKERS, shuffle=True,
    )

    if debug:
        val_set = data.Subset(val_set, range(3))

    val_loader = data.DataLoader(val_set, batch_size=config.VALIDATION.BATCH_SIZE_PER_GPU, num_workers=config.WORKERS)

    model = getattr(models, config.MODEL.NAME).get_seg_model(config)

    model = model.to(device)  # Send to GPU

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.TRAIN.MAX_LR,
        momentum=config.TRAIN.MOMENTUM,
        weight_decay=config.TRAIN.WEIGHT_DECAY,
    )

    try:
        output_dir = generate_path(
            config.OUTPUT_DIR, git_branch(), git_hash(), config_file_name, config.TRAIN.MODEL_DIR, current_datetime(),
        )
    except TypeError:
        output_dir = generate_path(config.OUTPUT_DIR, config_file_name, config.TRAIN.MODEL_DIR, current_datetime(),)

    summary_writer = create_summary_writer(log_dir=path.join(output_dir, config.LOG_DIR))
    snapshot_duration = epochs_per_cycle * len(train_loader) if not debug else 2 * len(train_loader)
    scheduler = CosineAnnealingScheduler(
        optimizer, "lr", config.TRAIN.MAX_LR, config.TRAIN.MIN_LR, cycle_size=snapshot_duration
    )

    # weights are inversely proportional to the frequency of the classes in
    # the training set
    class_weights = torch.tensor(config.DATASET.CLASS_WEIGHTS, device=device, requires_grad=False)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=mask_value, reduction="mean")

    trainer = create_supervised_trainer(model, optimizer, criterion, _prepare_batch, device=device)

    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, logging_handlers.log_training_output(log_interval=config.TRAIN.BATCH_SIZE_PER_GPU),
    )
    trainer.add_event_handler(Events.EPOCH_STARTED, logging_handlers.log_lr(optimizer))
    trainer.add_event_handler(
        Events.EPOCH_STARTED, tensorboard_handlers.log_lr(summary_writer, optimizer, "epoch"),
    )
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, tensorboard_handlers.log_training_output(summary_writer),
    )

    def _select_pred_and_mask(model_out_dict):
        return (model_out_dict["y_pred"].squeeze(), model_out_dict["mask"].squeeze())

    evaluator = create_supervised_evaluator(
        model,
        _prepare_batch,
        metrics={
            "pixacc": pixelwise_accuracy(n_classes, output_transform=_select_pred_and_mask),
            "nll": Loss(criterion, output_transform=_select_pred_and_mask),
            "cacc": class_accuracy(n_classes, output_transform=_select_pred_and_mask),
            "mca": mean_class_accuracy(n_classes, output_transform=_select_pred_and_mask),
            "ciou": class_iou(n_classes, output_transform=_select_pred_and_mask),
            "mIoU": mean_iou(n_classes, output_transform=_select_pred_and_mask),
        },
        device=device,
    )

    # Set the validation run to start on the epoch completion of the training run
    trainer.add_event_handler(Events.EPOCH_COMPLETED, Evaluator(evaluator, val_loader))

    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        logging_handlers.log_metrics(
            "Validation results",
            metrics_dict={
                "nll": "Avg loss :",
                "pixacc": "Pixelwise Accuracy :",
                "mca": "Avg Class Accuracy :",
                "mIoU": "Avg Class IoU :",
            },
        ),
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        tensorboard_handlers.log_metrics(
            summary_writer,
            trainer,
            "epoch",
            metrics_dict={
                "mIoU": "Validation/mIoU",
                "nll": "Validation/Loss",
                "mca": "Validation/MCA",
                "pixacc": "Validation/Pixel_Acc",
            },
        ),
    )

    def _select_max(pred_tensor):
        return pred_tensor.max(1)[1]

    def _tensor_to_numpy(pred_tensor):
        return pred_tensor.squeeze().cpu().numpy()

    transform_func = compose(np_to_tb, decode_segmap, _tensor_to_numpy,)

    transform_pred = compose(transform_func, _select_max)

    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED, create_image_writer(summary_writer, "Validation/Image", "image"),
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        create_image_writer(summary_writer, "Validation/Mask", "mask", transform_func=transform_func),
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        create_image_writer(summary_writer, "Validation/Pred", "y_pred", transform_func=transform_pred),
    )

    def snapshot_function():
        return (trainer.state.iteration % snapshot_duration) == 0

    checkpoint_handler = SnapshotHandler(output_dir, config.MODEL.NAME, extract_metric_from("mIoU"), snapshot_function,)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {"model": model})

    logger.info("Starting training")
    if debug:
        trainer.run(
            train_loader,
            max_epochs=config.TRAIN.END_EPOCH,
            epoch_length=config.TRAIN.BATCH_SIZE_PER_GPU,
            seed=config.SEED,
        )
    else:
        trainer.run(train_loader, max_epochs=config.TRAIN.END_EPOCH, epoch_length=len(train_loader), seed=config.SEED)


if __name__ == "__main__":
    fire.Fire(run)
