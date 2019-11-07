# Copyright (c) Microsoft Corporation. All rights reserved.
# # Licensed under the MIT License.
# # /* spell-checker: disable */

import logging
import logging.config
from os import path

import fire
import numpy as np
import torch
from albumentations import Compose, HorizontalFlip, Normalize
from deepseismic_interpretation.dutchf3.data import (
    decode_segmap,
    get_section_loader,
)
from cv_lib.event_handlers import (
    SnapshotHandler,
    logging_handlers,
    tensorboard_handlers,
)
from cv_lib.event_handlers.logging_handlers import Evaluator
from cv_lib.event_handlers.tensorboard_handlers import (
    create_image_writer,
    create_summary_writer,
)
from cv_lib.segmentation import models
from cv_lib.segmentation.dutchf3.engine import (
    create_supervised_evaluator,
    create_supervised_trainer,
)
from cv_lib.segmentation.dutchf3.metrics import (
    FrequencyWeightedIoU,
    MeanClassAccuracy,
    MeanIoU,
    PixelwiseAccuracy,
)
from cv_lib.segmentation.dutchf3.utils import (
    current_datetime,
    generate_path,
    git_branch,
    git_hash,
    np_to_tb,
)
from default import _C as config
from default import update_config
from ignite.contrib.handlers import CosineAnnealingScheduler
from ignite.engine import Events
from ignite.metrics import Loss
from ignite.utils import convert_tensor
from toolz import compose
from torch.utils import data

from cv_lib.segmentation import extract_metric_from


def prepare_batch(batch, device="cuda", non_blocking=False):
    x, y = batch
    return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )


def run(*options, cfg=None):
    """Run training and validation of model

    Notes:
        Options can be passed in via the options argument and loaded from the cfg file
        Options loaded from default.py will be overridden by options loaded from cfg file
        Options passed in through options argument will override option loaded from cfg file

    Args:
        *options (str,int ,optional): Options used to overide what is loaded from the config.
                                      To see what options are available consult default.py
        cfg (str, optional): Location of config file to load. Defaults to None.
    """

    update_config(config, options=options, config_file=cfg)
    logging.config.fileConfig(config.LOG_CONFIG)
    logger = logging.getLogger(__name__)
    logger.debug(config.WORKERS)
    scheduler_step = config.TRAIN.END_EPOCH // config.TRAIN.SNAPSHOTS
    torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK

    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(seed=config.SEED)

    # Setup Augmentations
    basic_aug = Compose(
        [
            Normalize(
                mean=(config.TRAIN.MEAN,),
                std=(config.TRAIN.STD,),
                max_pixel_value=1,
            )
        ]
    )
    if config.TRAIN.AUGMENTATION:
        train_aug = Compose([basic_aug, HorizontalFlip(p=0.5)])
        val_aug = basic_aug
    else:
        train_aug = val_aug = basic_aug

    TrainLoader = get_section_loader(config)

    train_set = TrainLoader(
        data_dir=config.DATASET.ROOT,
        split="train",
        is_transform=True,
        augmentations=train_aug,
    )

    val_set = TrainLoader(
        data_dir=config.DATASET.ROOT,
        split="val",
        is_transform=True,
        augmentations=val_aug,
    )

    class CustomSampler(torch.utils.data.Sampler):
        def __init__(self, data_source):
            self.data_source = data_source

        def __iter__(self):
            char = ["i" if np.random.randint(2) == 1 else "x"]
            self.indices = [
                idx
                for (idx, name) in enumerate(self.data_source)
                if char[0] in name
            ]
            return (self.indices[i] for i in torch.randperm(len(self.indices)))

        def __len__(self):
            return len(self.data_source)

    n_classes = train_set.n_classes

    val_list = val_set.sections
    train_list = val_set.sections

    train_loader = data.DataLoader(
        train_set,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        sampler=CustomSampler(train_list),
        num_workers=config.WORKERS,
        shuffle=False,
    )

    val_loader = data.DataLoader(
        val_set,
        batch_size=config.VALIDATION.BATCH_SIZE_PER_GPU,
        sampler=CustomSampler(val_list),
        num_workers=config.WORKERS,
    )

    model = getattr(models, config.MODEL.NAME).get_seg_model(config)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    model = model.to(device)  # Send to GPU

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.TRAIN.MAX_LR,
        momentum=config.TRAIN.MOMENTUM,
        weight_decay=config.TRAIN.WEIGHT_DECAY,
    )

    output_dir = generate_path(
        config.OUTPUT_DIR,
        git_branch(),
        git_hash(),
        config.MODEL.NAME,
        current_datetime(),
    )

    summary_writer = create_summary_writer(
        log_dir=path.join(output_dir, config.LOG_DIR)
    )

    snapshot_duration = scheduler_step * len(train_loader)
    scheduler = CosineAnnealingScheduler(
        optimizer,
        "lr",
        config.TRAIN.MAX_LR,
        config.TRAIN.MIN_LR,
        snapshot_duration,
    )

    # weights are inversely proportional to the frequency of the classes in the training set
    class_weights = torch.tensor(
        config.DATASET.CLASS_WEIGHTS, device=device, requires_grad=False
    )

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255, reduction="mean"
    )

    trainer = create_supervised_trainer(
        model, optimizer, criterion, prepare_batch, device=device
    )

    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED,
        logging_handlers.log_training_output(log_interval=config.PRINT_FREQ),
    )

    trainer.add_event_handler(
        Events.EPOCH_STARTED, logging_handlers.log_lr(optimizer)
    )

    trainer.add_event_handler(
        Events.EPOCH_STARTED,
        tensorboard_handlers.log_lr(summary_writer, optimizer, "epoch"),
    )

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED,
        tensorboard_handlers.log_training_output(summary_writer),
    )

    def _select_pred_and_mask(model_out_dict):
        return (
            model_out_dict["y_pred"].squeeze(),
            model_out_dict["mask"].squeeze(),
        )

    evaluator = create_supervised_evaluator(
        model,
        prepare_batch,
        metrics={
            "IoU": MeanIoU(
                n_classes, device, output_transform=_select_pred_and_mask
            ),
            "nll": Loss(criterion, output_transform=_select_pred_and_mask),
            "mca": MeanClassAccuracy(
                n_classes, device, output_transform=_select_pred_and_mask
            ),
            "fiou": FrequencyWeightedIoU(
                n_classes, device, output_transform=_select_pred_and_mask
            ),
            "pixa": PixelwiseAccuracy(
                n_classes, device, output_transform=_select_pred_and_mask
            ),
        },
        device=device,
    )

    # Set the validation run to start on the epoch completion of the training run
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, Evaluator(evaluator, val_loader)
    )

    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        logging_handlers.log_metrics(
            "Validation results",
            metrics_dict={
                "IoU": "IoU :",
                "nll": "Avg loss :",
                "pixa": "Pixelwise Accuracy :",
                "mca": "Mean Class Accuracy :",
                "fiou": "Freq Weighted IoU :",
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
                "IoU": "Validation/IoU",
                "nll": "Validation/Loss",
                "mca": "Validation/MCA",
                "fiou": "Validation/FIoU",
            },
        ),
    )

    def _select_max(pred_tensor):
        return pred_tensor.max(1)[1]

    def _tensor_to_numpy(pred_tensor):
        return pred_tensor.squeeze().cpu().numpy()

    transform_func = compose(
        np_to_tb, decode_segmap(n_classes=n_classes), _tensor_to_numpy
    )

    transform_pred = compose(transform_func, _select_max)

    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        create_image_writer(summary_writer, "Validation/Image", "image"),
    )

    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        create_image_writer(
            summary_writer,
            "Validation/Mask",
            "mask",
            transform_func=transform_func,
        ),
    )

    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        create_image_writer(
            summary_writer,
            "Validation/Pred",
            "y_pred",
            transform_func=transform_pred,
        ),
    )

    def snapshot_function():
        return (trainer.state.iteration % snapshot_duration) == 0

    checkpoint_handler = SnapshotHandler(
        path.join(output_dir, config.TRAIN.MODEL_DIR),
        config.MODEL.NAME,
        extract_metric_from("fiou"),
        snapshot_function,
    )

    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED, checkpoint_handler, {"model": model}
    )

    logger.info("Starting training")
    trainer.run(train_loader, max_epochs=config.TRAIN.END_EPOCH)


if __name__ == "__main__":
    fire.Fire(run)
