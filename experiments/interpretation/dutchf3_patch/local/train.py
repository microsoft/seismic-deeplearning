# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# To Test:
# python train.py TRAIN.END_EPOCH 1 TRAIN.SNAPSHOTS 1 --cfg "configs/hrnet.yaml" --debug
#
# /* spell-checker: disable */
"""Train models on Dutch F3 dataset

Trains models using PyTorch
Uses a warmup schedule that then goes into a cyclic learning rate

Time to run on single V100 for 300 epochs: 4.5 days
"""
import json
import logging
import logging.config
from os import path

import fire
import numpy as np
import torch
from torch.utils import data
from albumentations import Compose, HorizontalFlip, Normalize, PadIfNeeded, Resize
from ignite.contrib.handlers import CosineAnnealingScheduler
from ignite.engine import Events
from ignite.metrics import Loss
from ignite.utils import convert_tensor

from cv_lib.event_handlers import SnapshotHandler, logging_handlers, tensorboard_handlers
from cv_lib.event_handlers.tensorboard_handlers import create_summary_writer
from cv_lib.segmentation import extract_metric_from, models
from cv_lib.segmentation.dutchf3.engine import create_supervised_evaluator, create_supervised_trainer
from cv_lib.segmentation.dutchf3.utils import current_datetime, generate_path, git_branch, git_hash
from cv_lib.segmentation.metrics import class_accuracy, class_iou, mean_class_accuracy, mean_iou, pixelwise_accuracy
from cv_lib.utils import load_log_configuration
from deepseismic_interpretation.dutchf3.data import get_patch_loader
from default import _C as config
from default import update_config


def prepare_batch(batch, device=None, non_blocking=False):
    x, y = batch
    return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )


def run(*options, cfg=None, debug=False):
    """Run training and validation of model

    Notes:
        Options can be passed in via the options argument and loaded from the cfg file
        Options from default.py will be overridden by options loaded from cfg file
        Options from default.py will be overridden by options loaded from cfg file
        Options passed in via options argument will override option loaded from cfg file
    
    Args:
        *options (str,int ,optional): Options used to overide what is loaded from the
                                      config. To see what options are available consult
                                      default.py
        cfg (str, optional): Location of config file to load. Defaults to None.        
        debug (bool): Places scripts in debug/test mode and only executes a few iterations
    """
    # Configuration:
    update_config(config, options=options, config_file=cfg)

    # The model will be saved under: outputs/<config_file_name>/<model_dir>
    config_file_name = "default_config" if not cfg else cfg.split("/")[-1].split(".")[0]
    try:
        output_dir = generate_path(
            config.OUTPUT_DIR, git_branch(), git_hash(), config_file_name, config.TRAIN.MODEL_DIR, current_datetime(),
        )
    except TypeError:
        output_dir = generate_path(config.OUTPUT_DIR, config_file_name, config.TRAIN.MODEL_DIR, current_datetime(),)    

    # Logging:
    load_log_configuration(config.LOG_CONFIG)
    logger = logging.getLogger(__name__)
    logger.debug(config.WORKERS)

    # Set CUDNN benchmark mode:
    torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK

    # we will write the model under outputs / config_file_name / model_dir
    config_file_name = "default_config" if not cfg else cfg.split("/")[-1].split(".")[0]

    # Fix random seeds:
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(seed=config.SEED)

    # Augmentation:
    basic_aug = Compose(
        [
            Normalize(mean=(config.TRAIN.MEAN,), std=(config.TRAIN.STD,), max_pixel_value=1),
            PadIfNeeded(
                min_height=config.TRAIN.PATCH_SIZE,
                min_width=config.TRAIN.PATCH_SIZE,
                border_mode=config.OPENCV_BORDER_CONSTANT,
                always_apply=True,
                mask_value=255,
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
                mask_value=255,
            ),
        ]
    )
    if config.TRAIN.AUGMENTATION:
        train_aug = Compose([basic_aug, HorizontalFlip(p=0.5)])
        val_aug = basic_aug
    else:
        train_aug = val_aug = basic_aug

    # Training and Validation Loaders:
    TrainPatchLoader = get_patch_loader(config)
    train_set = TrainPatchLoader(
        config.DATASET.ROOT,
        config.DATASET.NUM_CLASSES,
        split="train",
        is_transform=True,
        stride=config.TRAIN.STRIDE,
        patch_size=config.TRAIN.PATCH_SIZE,
        augmentations=train_aug
    )
    logger.info(train_set)
    n_classes = train_set.n_classes
    val_set = TrainPatchLoader(
        config.DATASET.ROOT,
        config.DATASET.NUM_CLASSES,
        split="val",
        is_transform=True,
        stride=config.TRAIN.STRIDE,
        patch_size=config.TRAIN.PATCH_SIZE,
        augmentations=val_aug,
    )
    logger.info(val_set)

    if debug:
        logger.info("Running in debug mode..")
        train_set = data.Subset(train_set, range(config.TRAIN.BATCH_SIZE_PER_GPU*config.NUM_DEBUG_BATCHES))
        val_set = data.Subset(val_set, range(config.VALIDATION.BATCH_SIZE_PER_GPU))

    train_loader = data.DataLoader(
        train_set, batch_size=config.TRAIN.BATCH_SIZE_PER_GPU, num_workers=config.WORKERS, shuffle=True
    )
    val_loader = data.DataLoader(
        val_set, batch_size=config.VALIDATION.BATCH_SIZE_PER_GPU, num_workers=1
    )  # config.WORKERS)

    # Model:
    model = getattr(models, config.MODEL.NAME).get_seg_model(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Optimizer and LR Scheduler:
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.TRAIN.MAX_LR,
        momentum=config.TRAIN.MOMENTUM,
        weight_decay=config.TRAIN.WEIGHT_DECAY,
    )

    epochs_per_cycle = config.TRAIN.END_EPOCH // config.TRAIN.SNAPSHOTS
    snapshot_duration = epochs_per_cycle * len(train_loader) if not debug else 2 * len(train_loader)
    scheduler = CosineAnnealingScheduler(
        optimizer, "lr", config.TRAIN.MAX_LR, config.TRAIN.MIN_LR, cycle_size=snapshot_duration
    )

    # Tensorboard writer:
    summary_writer = create_summary_writer(log_dir=path.join(output_dir, "logs"))

    # class weights are inversely proportional to the frequency of the classes in the training set
    class_weights = torch.tensor(config.DATASET.CLASS_WEIGHTS, device=device, requires_grad=False)

    # Loss:
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, reduction="mean")

    # Ignite trainer and evaluator:
    trainer = create_supervised_trainer(model, optimizer, criterion, prepare_batch, device=device)
    transform_fn = lambda output_dict: (output_dict["y_pred"].squeeze(), output_dict["mask"].squeeze())
    evaluator = create_supervised_evaluator(
        model,
        prepare_batch,
        metrics={
            "nll": Loss(criterion, output_transform=transform_fn),
            "pixacc": pixelwise_accuracy(n_classes, output_transform=transform_fn, device=device),
            "cacc": class_accuracy(n_classes, output_transform=transform_fn),
            "mca": mean_class_accuracy(n_classes, output_transform=transform_fn),
            "ciou": class_iou(n_classes, output_transform=transform_fn),
            "mIoU": mean_iou(n_classes, output_transform=transform_fn),
        },
        device=device,
    )
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Logging:
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, logging_handlers.log_training_output(log_interval=config.PRINT_FREQ),
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, logging_handlers.log_lr(optimizer))

    # Tensorboard and Logging:
    trainer.add_event_handler(Events.ITERATION_COMPLETED, tensorboard_handlers.log_training_output(summary_writer))
    trainer.add_event_handler(Events.ITERATION_COMPLETED, tensorboard_handlers.log_validation_output(summary_writer))

    # add specific logger which also triggers printed metrics on training set
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        tensorboard_handlers.log_results(engine, evaluator, summary_writer, n_classes, stage="Training")
        logging_handlers.log_metrics(engine, evaluator, stage="Training")

    # add specific logger which also triggers printed metrics on validation set
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        tensorboard_handlers.log_results(engine, evaluator, summary_writer, n_classes, stage="Validation")
        logging_handlers.log_metrics(engine, evaluator, stage="Validation")
        # dump validation set metrics at the very end for debugging purposes
        if engine.state.epoch == config.TRAIN.END_EPOCH and debug:
            fname = f"metrics_test_{config_file_name}_{config.TRAIN.MODEL_DIR}.json"
            metrics = evaluator.state.metrics
            out_dict = {x: metrics[x] for x in ["nll", "pixacc", "mca", "mIoU"]}
            with open(fname, "w") as fid:
                json.dump(out_dict, fid)
            log_msg = " ".join(f"{k}: {out_dict[k]}" for k in out_dict.keys())
            logging.info(log_msg)

    # Checkpointing: snapshotting trained models to disk
    checkpoint_handler = SnapshotHandler(
        output_dir,
        config.MODEL.NAME,
        extract_metric_from("mIoU"),
        lambda: (trainer.state.iteration % snapshot_duration) == 0,
    )
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {"model": model})

    logger.info("Starting training")
    trainer.run(train_loader, max_epochs=config.TRAIN.END_EPOCH, epoch_length=len(train_loader), seed=config.SEED)

    summary_writer.close()


if __name__ == "__main__":
    fire.Fire(run)
