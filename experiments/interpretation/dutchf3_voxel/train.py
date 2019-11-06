# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# /* spell-checker: disable */

import logging
import logging.config
from os import path

import fire
import numpy as np
import torch
from torch.utils import data
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss
# TODO: get mertircs from Ignite
# from ignite.metrics import MIoU, MeanClassAccuracy, FrequencyWeightedIoU, PixelwiseAccuracy
from ignite.utils import convert_tensor
from ignite.engine.engine import Engine
from toolz import compose, curry
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

# TODO: replace with Ignite metrics
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


def _prepare_batch(
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

    trainer = create_supervised_trainer(
        model,
        optimizer,
        loss,
        prepare_batch=_prepare_batch,
        device=device,
    )

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader), desc=desc.format(0)
    )

    # add model checkpointing
    output_dir = path.join(config.OUTPUT_DIR, config.TRAIN.MODEL_DIR)
    checkpoint_handler = ModelCheckpoint(
        output_dir, "model", save_interval=1, n_saved=3, create_dir=True, require_empty=False)

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    
    # save model at each epoch
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, checkpoint_handler, {config.MODEL.NAME: model}
    )
    
    def _select_pred_and_mask(model_out):
        # receive a tuple of (x, y_pred), y
        # so actually in line 51 of cv_lib/cv_lib/segmentation/dutch_f3/metrics/__init__.py
        # we do the following line, so here we just select the model        
        #_, y_pred = torch.max(model_out[0].squeeze(), 1, keepdim=True)
        y_pred = model_out[0].squeeze()
        y = model_out[1].squeeze()
        return (y_pred.squeeze(), y,)
    
    evaluator = create_supervised_evaluator(
        model,        
        metrics={
            "IoU": MeanIoU(
                n_classes, device, output_transform=_select_pred_and_mask
            ),
            "nll": Loss(criterion),
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
        prepare_batch=_prepare_batch,
    )    

    # Set the validation run to start on the epoch completion of the training run
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, Evaluator(evaluator, val_loader)
    )

    summary_writer = create_summary_writer(
        log_dir=path.join(output_dir, config.LOG_DIR)
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
    
    summary_writer = create_summary_writer(
        log_dir=path.join(output_dir, config.LOG_DIR)
    )   
    

    snapshot_duration=1
    def snapshot_function():
        return (trainer.state.iteration % snapshot_duration) == 0

    checkpoint_handler = SnapshotHandler(
        path.join(output_dir, config.TRAIN.MODEL_DIR),
        config.MODEL.NAME,
        snapshot_function,
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED, checkpoint_handler, {"model": model}
    )

    logger.info("Starting training")
    trainer.run(train_loader, max_epochs=config.TRAIN.END_EPOCH//config.TRAIN.BATCH_PER_EPOCH)
    pbar.close()

if __name__ == "__main__":
    fire.Fire(run)

