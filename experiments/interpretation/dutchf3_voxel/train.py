# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# /* spell-checker: disable */

import logging
import logging.config
from os import path

import cv2
import fire
import numpy as np
import torch
from albumentations import (
    Compose,
    HorizontalFlip,
    Normalize,
    PadIfNeeded,
    Resize,
)
from ignite.contrib.handlers import CosineAnnealingScheduler
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss
from ignite.utils import convert_tensor
from toolz import compose
from torch.utils import data
from tqdm import tqdm

from deepseismic_interpretation.dutchf3.data import (
    get_voxel_loader,
    decode_segmap,
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

# from cv_lib.segmentation.dutchf3.engine import (
#    create_supervised_evaluator,
#    create_supervised_trainer,
# )
# Use ignite generic versions for now
from ignite.engine import (
    create_supervised_trainer,
    create_supervised_evaluator,
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

from interpretation.deepseismic_interpretation.models.texture_net import (
    TextureNet,
)

from default import _C as config
from default import update_config


def prepare_batch(
    batch, device=None, non_blocking=False, t_type=torch.FloatTensor
):
    x, y = batch
    new_x = convert_tensor(
        torch.squeeze(x, 1), device=device, non_blocking=non_blocking
    )
    new_y = convert_tensor(torch.unsqueeze(y, 2), device=device, non_blocking=non_blocking)
    if device == "cuda":
        return new_x.type(t_type).cuda(), torch.unsqueeze(new_y, 3).type(torch.LongTensor).cuda()
    else:
        return new_x.type(t_type), torch.unsqueeze(new_y, 3).type(torch.LongTensor)


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
    torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK

    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(seed=config.SEED)

    # load the data
    TrainVoxelLoader = get_voxel_loader(config)

    train_set = TrainVoxelLoader(
        config.DATASET.ROOT,
        config.DATASET.FILENAME,
        split="train",
        window_size=config.WINDOW_SIZE,
        len=config.TRAIN.BATCH_SIZE_PER_GPU*config.TRAIN.BATCH_PER_EPOCH,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
    )
    val_set = TrainVoxelLoader(
        config.DATASET.ROOT,
        config.DATASET.FILENAME,
        split="val",
        window_size=config.WINDOW_SIZE,
        len=config.TRAIN.BATCH_SIZE_PER_GPU*config.TRAIN.BATCH_PER_EPOCH,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
    )

    n_classes = train_set.n_classes

    # set dataset length to batch size to be consistent with 5000 iterations each of size 
    # 32 in the original Waldeland implementation
    train_loader = data.DataLoader(
        train_set,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        num_workers=config.WORKERS,        
        shuffle=False        
    )
    val_loader = data.DataLoader(
        val_set,
        batch_size=config.VALIDATION.BATCH_SIZE_PER_GPU,
        num_workers=config.WORKERS,
        shuffle=False      
    )

    # this is how we import model for CV - here we're importing a seismic segmentation model
    # model = getattr(models, config.MODEL.NAME).get_seg_model(config)
    # TODO: pass more model parameters into the mode from config
    model = TextureNet(n_classes=config.DATASET.NUM_CLASSES)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.TRAIN.LR,
        # momentum=config.TRAIN.MOMENTUM,
        weight_decay=config.TRAIN.WEIGHT_DECAY,
    )

    device = "cpu"
    log_interval = 10
    if torch.cuda.is_available():
        device = "cuda"
        model = model.cuda()

    loss = torch.nn.CrossEntropyLoss()

    def _select_pred_and_mask(model_out_dict):
        return (
            model_out_dict["y_pred"].squeeze(),
            model_out_dict["mask"].squeeze(),
        )

    trainer = create_supervised_trainer(
        model,
        optimizer,
        loss,
        prepare_batch=prepare_batch,        
        device=device,
    )

    evaluator = create_supervised_evaluator(
        model,
        prepare_batch=prepare_batch,
        metrics={
            "accuracy": Accuracy(),
            "nll": Loss(loss),
        },
        device=device,
    )

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader), desc=desc.format(0)
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_interval == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(log_interval)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_loss = metrics["nll"]
        tqdm.write(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                engine.state.epoch, avg_accuracy, avg_loss
            )
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_loss = metrics["nll"]
        tqdm.write(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                engine.state.epoch, avg_accuracy, avg_loss
            )
        )

        pbar.n = pbar.last_print_n = 0

    trainer.run(train_loader, max_epochs=config.TRAIN.END_EPOCH//config.TRAIN.BATCH_PER_EPOCH)
    pbar.close()

if __name__ == "__main__":
    fire.Fire(run)

