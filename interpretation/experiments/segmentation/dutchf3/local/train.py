import logging
import logging.config
import fire
import os
from datetime import datetime

import itertools

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch.utils import data
from tqdm import tqdm
from default import _C as config
from default import update_config
from loss import cross_entropy
import torchvision.utils as vutils
from augmentations import (
    Compose,
    RandomHorizontallyFlip,
    RandomRotate,
    AddNoise,
)
from loader.data_loader import TrainPatchLoader
from metrics import runningScore
from models import get_model
from utils import np_to_tb
from data import split_train_val
from ignite.contrib.handlers import CosineAnnealingScheduler
from cv_lib.segmentation.dutchf3.engine import (
    create_supervised_trainer,
)
from ignite.utils import convert_tensor
from ignite.engine import Events
from cv_lib.event_handlers import (
    SnapshotHandler,
    logging_handlers,
    tensorboard_handlers,
)
from models import get_model

def prepare_train_batch(batch, device=None, non_blocking=False):
    x, y = batch
    return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )


def prepare_val_batch(batch, device=None, non_blocking=False):
    x, y = batch
    return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )

def run(*options, cfg=None):
    fraction_validation=0.2
    update_config(config, options=options, config_file=cfg)
    print(config.LOG_CONFIG)
    logging.config.fileConfig(config.LOG_CONFIG)
    scheduler_step = config.TRAIN.END_EPOCH // config.TRAIN.SNAPSHOTS
    torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK

    torch.manual_seed(2019)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2019)
    np.random.seed(seed=2019)

    # Generate the train and validation sets for the model:
    split_train_val(config.DATASET.STRIDE, per_val=fraction_validation, loader_type="patch")

    # Setup Augmentations
    if config.DATASET.AUGMENTATION:
        data_aug = Compose(
            [RandomRotate(10), RandomHorizontallyFlip(), AddNoise()]
        )
    else:
        data_aug = None

    train_set = TrainPatchLoader(
        split="train",
        is_transform=True,
        stride=config.DATASET.STRIDE,
        patch_size=config.DATASET.PATCH_SIZE,
        augmentations=data_aug,
    )

    # Without Augmentation:
    val_set = TrainPatchLoader(
        split="train",
        is_transform=True,
        stride=config.DATASET.STRIDE,
        patch_size=config.DATASET.PATCH_SIZE,
    )

    n_classes = train_set.n_classes

    train_loader = data.DataLoader(
        train_set, batch_size=config.TRAIN.BATCH_SIZE_PER_GPU, num_workers=0, shuffle=True
    )
    val_loader = data.DataLoader(
        train_set, batch_size=config.TRAIN.BATCH_SIZE_PER_GPU, num_workers=0
    )
    
    model = get_model("patch_deconvnet", False, n_classes)

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

    # summary_writer = create_summary_writer(log_dir=config.LOG_DIR)
    snapshot_duration = scheduler_step * len(train_set)
    scheduler = CosineAnnealingScheduler(
        optimizer, "lr", config.TRAIN.MAX_LR, config.TRAIN.MIN_LR, snapshot_duration
    )

    # weights are inversely proportional to the frequency of the classes in the training set
    class_weights = torch.tensor(
        [0.7151, 0.8811, 0.5156, 0.9346, 0.9683, 0.9852],
        device=device,
        requires_grad=False,
    )
   
    criterion = cross_entropy(weight=class_weights)

    trainer = create_supervised_trainer(
        model, optimizer, criterion, prepare_train_batch, device=device
    )

    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED,
        logging_handlers.log_training_output(log_interval=config.PRINT_FREQ),
    )
    trainer.add_event_handler(Events.EPOCH_STARTED, logging_handlers.log_lr(optimizer))
  
    logger = logging.getLogger(__name__)
    logger.info("Starting training")
    trainer.run(train_loader, max_epochs=config.TRAIN.END_EPOCH)

if __name__ == "__main__":
    fire.Fire(run)
