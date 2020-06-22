# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# To Test:
# python train.py TRAIN.END_EPOCH 1 TRAIN.SNAPSHOTS 1 --cfg "configs/seresnet_unet.yaml" --debug
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

import os
from os import path

import fire
import numpy as np
import torch
from albumentations import Compose, HorizontalFlip, Normalize, PadIfNeeded, Resize
from ignite.contrib.handlers import ConcatScheduler, CosineAnnealingScheduler, LinearCyclicalScheduler
from ignite.engine import Events
from ignite.metrics import Loss
from ignite.utils import convert_tensor
from toolz import curry
from torch.utils import data

from cv_lib.event_handlers import SnapshotHandler, logging_handlers, tensorboard_handlers
from cv_lib.event_handlers.tensorboard_handlers import create_summary_writer
from cv_lib.segmentation import extract_metric_from, models
from cv_lib.segmentation.dutchf3.engine import create_supervised_evaluator, create_supervised_trainer
from cv_lib.segmentation.dutchf3.utils import current_datetime, git_branch, git_hash
from cv_lib.segmentation.metrics import class_accuracy, class_iou, mean_class_accuracy, mean_iou, pixelwise_accuracy
from cv_lib.utils import generate_path, load_log_configuration
from deepseismic_interpretation.dutchf3.data import get_patch_loader
from default import _C as config
from default import update_config


def prepare_batch(batch, device=None, non_blocking=False):
    x, y = batch
    return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )


@curry
def update_sampler_epoch(data_loader, engine):
    data_loader.sampler.epoch = engine.state.epoch


def run(*options, cfg=None, local_rank=0, debug=False, input=None, distributed=False):
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
        input (str, optional): Location of data if Azure ML run, 
            for local runs input is config.DATASET.ROOT
        distributed (bool): This flag tells the training script to run in distributed mode
            if more than one GPU exists.
    """

    # if AML training pipeline supplies us with input
    if input is not None:
        data_dir = input
        output_dir = data_dir + config.OUTPUT_DIR

    # Start logging
    load_log_configuration(config.LOG_CONFIG)
    logger = logging.getLogger(__name__)
    logger.debug(config.WORKERS)

    # Configuration:
    update_config(config, options=options, config_file=cfg)
    silence_other_ranks = True

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1

    if distributed:
        # FOR DISTRIBUTED: Set the device according to local_rank.
        torch.cuda.set_device(local_rank)

        # FOR DISTRIBUTED: Initialize the backend. torch.distributed.launch will
        # provide environment variables, and requires that you use init_method=`env://`.
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        logging.info(f"Started train.py using distributed mode.")
    else:
        logging.info(f"Started train.py using local mode.")

    # Set CUDNN benchmark mode:
    torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK

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
    logging.info(f"Using {TrainPatchLoader}")

    train_set = TrainPatchLoader(config, split="train", is_transform=True, augmentations=train_aug, debug=debug,)
    logger.info(train_set)

    n_classes = train_set.n_classes
    val_set = TrainPatchLoader(config, split="val", is_transform=True, augmentations=val_aug, debug=debug,)

    logger.info(val_set)

    if debug:
        data_flow_dict = dict()

        data_flow_dict["train_patch_loader_length"] = len(train_set)
        data_flow_dict["validation_patch_loader_length"] = len(val_set)
        data_flow_dict["train_input_shape"] = train_set.seismic.shape
        data_flow_dict["train_label_shape"] = train_set.labels.shape
        data_flow_dict["n_classes"] = n_classes

        logger.info("Running in debug mode..")
        train_range = min(config.TRAIN.BATCH_SIZE_PER_GPU * config.NUM_DEBUG_BATCHES, len(train_set))
        logging.info(f"train range in debug mode {train_range}")
        train_set = data.Subset(train_set, range(train_range))
        valid_range = min(config.VALIDATION.BATCH_SIZE_PER_GPU, len(val_set))
        val_set = data.Subset(val_set, range(valid_range))

        data_flow_dict["train_length_subset"] = len(train_set)
        data_flow_dict["validation_length_subset"] = len(val_set)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=local_rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, num_replicas=world_size, rank=local_rank)

    train_loader = data.DataLoader(
        train_set, batch_size=config.TRAIN.BATCH_SIZE_PER_GPU, num_workers=config.WORKERS, sampler=train_sampler,
    )
    val_loader = data.DataLoader(
        val_set, batch_size=config.VALIDATION.BATCH_SIZE_PER_GPU, num_workers=config.WORKERS, sampler=val_sampler
    )

    if debug:
        data_flow_dict["train_loader_length"] = len(train_loader)
        data_flow_dict["validation_loader_length"] = len(val_loader)
        config_file_name = "default_config" if not cfg else cfg.split("/")[-1].split(".")[0]
        fname = f"data_flow_train_{config_file_name}_{config.TRAIN.MODEL_DIR}.json"
        with open(fname, "w") as f:
            json.dump(data_flow_dict, f, indent=2)

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
    cosine_scheduler = CosineAnnealingScheduler(
        optimizer,
        "lr",
        config.TRAIN.MAX_LR * world_size,
        config.TRAIN.MIN_LR * world_size,
        cycle_size=snapshot_duration,
    )

    if distributed:
        warmup_duration = 5 * len(train_loader)
        warmup_scheduler = LinearCyclicalScheduler(
            optimizer,
            "lr",
            start_value=config.TRAIN.MAX_LR,
            end_value=config.TRAIN.MAX_LR * world_size,
            cycle_size=10 * len(train_loader),
        )
        scheduler = ConcatScheduler(schedulers=[warmup_scheduler, cosine_scheduler], durations=[warmup_duration])
    else:
        scheduler = cosine_scheduler

    # class weights are inversely proportional to the frequency of the classes in the training set
    class_weights = torch.tensor(config.DATASET.CLASS_WEIGHTS, device=device, requires_grad=False)

    # Loss:
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, reduction="mean")

    # Model:
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)
        if silence_other_ranks & local_rank != 0:
            logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)

    # Ignite trainer and evaluator:
    trainer = create_supervised_trainer(model, optimizer, criterion, prepare_batch, device=device)
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    # Set to update the epoch parameter of our distributed data sampler so that we get
    # different shuffles
    trainer.add_event_handler(Events.EPOCH_STARTED, update_sampler_epoch(train_loader))

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

    # The model will be saved under: outputs/<config_file_name>/<model_dir>
    config_file_name = "default_config" if not cfg else cfg.split("/")[-1].split(".")[0]
    try:
        output_dir = generate_path(
            config.OUTPUT_DIR, git_branch(), git_hash(), config_file_name, config.TRAIN.MODEL_DIR, current_datetime(),
        )
    except:
        output_dir = generate_path(config.OUTPUT_DIR, config_file_name, config.TRAIN.MODEL_DIR, current_datetime(),)

    if local_rank == 0:  # Run only on master process
        # Logging:
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED, logging_handlers.log_training_output(log_interval=config.PRINT_FREQ),
        )
        trainer.add_event_handler(Events.EPOCH_STARTED, logging_handlers.log_lr(optimizer))

        # Checkpointing: snapshotting trained models to disk
        checkpoint_handler = SnapshotHandler(
            output_dir,
            config.MODEL.NAME,
            extract_metric_from("mIoU"),
            lambda: (trainer.state.iteration % snapshot_duration) == 0,
        )

        evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {"model": model})

        # Tensorboard and Logging:
        summary_writer = create_summary_writer(log_dir=path.join(output_dir, "logs"))
        trainer.add_event_handler(Events.ITERATION_COMPLETED, tensorboard_handlers.log_training_output(summary_writer))
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED, tensorboard_handlers.log_validation_output(summary_writer)
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        if local_rank == 0:  # Run only on master process
            tensorboard_handlers.log_results(engine, evaluator, summary_writer, n_classes, stage="Training")
            logging_handlers.log_metrics(engine, evaluator, stage="Training")
            logger.info("Logging training results..")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        if local_rank == 0:  # Run only on master process
            tensorboard_handlers.log_results(engine, evaluator, summary_writer, n_classes, stage="Validation")
            logging_handlers.log_metrics(engine, evaluator, stage="Validation")
            logger.info("Logging validation results..")
            # dump validation set metrics at the very end for debugging purposes
            if engine.state.epoch == config.TRAIN.END_EPOCH and debug:
                fname = f"metrics_{config_file_name}_{config.TRAIN.MODEL_DIR}.json"
                metrics = evaluator.state.metrics
                out_dict = {x: metrics[x] for x in ["nll", "pixacc", "mca", "mIoU"]}
                with open(fname, "w") as fid:
                    json.dump(out_dict, fid)
                log_msg = " ".join(f"{k}: {out_dict[k]}" for k in out_dict.keys())
                logging.info(log_msg)

    logger.info("Starting training")
    trainer.run(train_loader, max_epochs=config.TRAIN.END_EPOCH, epoch_length=len(train_loader), seed=config.SEED)
    if local_rank == 0:
        summary_writer.close()


if __name__ == "__main__":
    fire.Fire(run)
