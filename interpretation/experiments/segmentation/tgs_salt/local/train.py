""" Train model on TGS Salt Dataset

"""

import logging
import logging.config

import fire
import torch
from default import _C as config
from default import update_config
from ignite.contrib.handlers import CosineAnnealingScheduler
from ignite.engine import Events
from ignite.metrics import Loss

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
    get_data_loaders,
    kfold_split,
    prepare_train_batch,
    prepare_val_batch,
)
    create_supervised_evaluator,
    create_supervised_trainer,
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
    scheduler_step = config.TRAIN.END_EPOCH // config.TRAIN.SNAPSHOTS
    torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK
    train_ids, _, _ = get_data_ids(
        config.DATASET.ROOT, train_csv="train.csv", depths_csv="depths.csv"
    )
    fold_generator = kfold_split(
        train_ids,
        n_splits=config.TEST.CV.N_SPLITS,
        random_state=config.TEST.CV.SEED,
        shuffle=config.TEST.CV.SHUFFLE,
    )
    train_idx, val_idx = next(fold_generator)
    val_ids = train_ids[val_idx]
    train_ids = train_ids[train_idx]
    train_loader, val_loader = get_data_loaders(
        train_ids,
        val_ids,
        config.TRAIN.BATCH_SIZE_PER_GPU,
        config.TEST.BATCH_SIZE_PER_GPU,
        config.TRAIN.FINE_SIZE,
        config.TRAIN.PAD_LEFT,
        config.TRAIN.PAD_RIGHT,
        config.DATASET.ROOT,
    )

    model = getattr(models, config.MODEL.NAME).get_seg_model(config)
    criterion = torch.nn.BCEWithLogitsLoss()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.TRAIN.MAX_LR,
        momentum=config.TRAIN.MOMENTUM,
        weight_decay=config.TRAIN.WEIGHT_DECAY,
    )

    summary_writer = create_summary_writer(log_dir=config.LOG_DIR)
    snapshot_duration = scheduler_step * len(train_loader)
    scheduler = CosineAnnealingScheduler(
        optimizer, "lr", config.TRAIN.MAX_LR, config.TRAIN.MIN_LR, snapshot_duration
    )

    trainer = create_supervised_trainer(
        model, optimizer, criterion, prepare_train_batch, device=device
    )

    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED,
        logging_handlers.log_training_output(log_interval=config.PRINT_FREQ),
    )
    trainer.add_event_handler(Events.EPOCH_STARTED, logging_handlers.log_lr(optimizer))
    trainer.add_event_handler(
        Events.EPOCH_STARTED,
        tensorboard_handlers.log_lr(summary_writer, optimizer, "epoch"),
    )
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED,
        tensorboard_handlers.log_training_output(summary_writer),
    )

    evaluator = create_supervised_evaluator(
        model,
        prepare_val_batch,
        metrics={
            "kaggle": KaggleMetric(output_transform=lambda x: (x["y_pred"], x["mask"])),
            "nll": Loss(criterion, output_transform=lambda x: (x["y_pred"], x["mask"])),
            "pixa": PixelwiseAccuracy(output_transform=lambda x: (x["y_pred"], x["mask"]))
        },
        device=device,
        output_transform=padded_val_transform(config.TRAIN.PAD_LEFT, config.TRAIN.FINE_SIZE),
    )

    # Set the validation run to start on the epoch completion of the training run
    trainer.add_event_handler(Events.EPOCH_COMPLETED, Evaluator(evaluator, val_loader))

    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        logging_handlers.log_metrics(
            "Validation results",
            metrics_dict={"kaggle": "Kaggle :", "nll": "Avg loss :", "pixa": "Pixelwise Accuracy :"},
        ),
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        tensorboard_handlers.log_metrics(
            summary_writer,
            trainer,
            "epoch",
            metrics_dict={"kaggle": "Validation/Kaggle", "nll": "Validation/Loss"},
        ),
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        create_image_writer(summary_writer, "Validation/Image", "image"),
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        create_image_writer(summary_writer, "Validation/Mask", "mask"),
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        create_image_writer(summary_writer, "Validation/Pred", "y_pred"),
    )

    def snapshot_function():
        return (trainer.state.iteration % snapshot_duration) == 0

    checkpoint_handler = SnapshotHandler(
        config.OUTPUT_DIR,
        config.MODEL.NAME,
        snapshot_function,
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED, checkpoint_handler, {"model": model}
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting training")
    trainer.run(train_loader, max_epochs=config.TRAIN.END_EPOCH)


if __name__ == "__main__":
    fire.Fire(run)
