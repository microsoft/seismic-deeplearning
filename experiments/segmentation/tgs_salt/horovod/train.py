"""Train models on TGS salt dataset

Trains models using PyTorch DistributedDataParallel
Uses a warmup schedule that then goes into a cyclic learning rate
Uses a weighted combination of Lovasz and BCE loss
"""

import logging
import logging.config

import fire
import horovod.torch as hvd
import torch
import torch.nn.functional as F
from default import _C as config
from default import update_config
from ignite.contrib.handlers import (
    CustomPeriodicEvent,
    CosineAnnealingScheduler,
    LinearCyclicalScheduler,
    ConcatScheduler,
)
from ignite.engine import Events
from toolz import curry

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
    get_distributed_data_loaders,
    kfold_split,
    prepare_train_batch,
    prepare_val_batch,
)
    create_supervised_evaluator,
    create_supervised_trainer,
)


@curry
def update_sampler_epoch(data_loader, engine):
    data_loader.sampler.epoch = engine.state.epoch


class CombinedLoss:
    """Creates a function that calculates weighted combined loss
    """

    def __init__(self, loss_functions, weights):
        """Initialise CombinedLoss

        Args:
            loss_functions (list): A list of PyTorch loss functions
            weights (list[int]): A list of weights to use when combining loss functions
        """
        self._losses = loss_functions
        self.weights = weights

    def __call__(self, input, target):
        # if weight is zero remove loss from calculations
        loss_functions_and_weights = filter(
            lambda l_w: l_w[1] > 0, zip(self._losses, self.weights)
        )
        loss_list = [
            weight * loss(input, target) for loss, weight in loss_functions_and_weights
        ]
        combined_loss = torch.stack(loss_list).sum()
        return combined_loss


@curry
def adjust_loss(loss_obj, weights, engine):
    loss_obj.weights = weights


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
    hvd.init()
    silence_other_ranks = True
    logging.config.fileConfig(config.LOG_CONFIG)

    torch.manual_seed(config.SEED)
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(config.SEED)
    rank, world_size = hvd.rank(), hvd.size()

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

    train_loader, val_loader = get_distributed_data_loaders(
        train_ids,
        val_ids,
        config.TRAIN.BATCH_SIZE_PER_GPU,
        config.TEST.BATCH_SIZE_PER_GPU,
        config.TRAIN.FINE_SIZE,
        config.TRAIN.PAD_LEFT,
        config.TRAIN.PAD_RIGHT,
        rank,
        world_size,
        config.DATASET.ROOT,
    )

    model = getattr(models, config.MODEL.NAME).get_seg_model(config)
    criterion = CombinedLoss(
        (lovasz_hinge, F.binary_cross_entropy_with_logits), config.LOSS.WEIGHTS
    )

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.TRAIN.MAX_LR,
        momentum=config.TRAIN.MOMENTUM,
        weight_decay=config.TRAIN.WEIGHT_DECAY,
    )

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if config.HOROVOD.FP16 else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=compression)

    summary_writer = create_summary_writer(log_dir=config.LOG_DIR)
    snapshot_duration = scheduler_step * len(train_loader)
    warmup_duration = 5 * len(train_loader)
    warmup_scheduler = LinearCyclicalScheduler(
        optimizer,
        "lr",
        start_value=config.TRAIN.MAX_LR,
        end_value=config.TRAIN.MAX_LR * world_size,
        cycle_size=10 * len(train_loader),
    )
    cosine_scheduler = CosineAnnealingScheduler(
        optimizer,
        "lr",
        config.TRAIN.MAX_LR * world_size,
        config.TRAIN.MIN_LR * world_size,
        snapshot_duration,
    )

    scheduler = ConcatScheduler(
        schedulers=[warmup_scheduler, cosine_scheduler], durations=[warmup_duration]
    )

    trainer = create_supervised_trainer(
        model, optimizer, criterion, prepare_train_batch, device=device
    )

    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    # Set to update the epoch parameter of our distributed data sampler so that we get different shuffles
    trainer.add_event_handler(Events.EPOCH_STARTED, update_sampler_epoch(train_loader))

    adjust_loss_event = CustomPeriodicEvent(n_epochs=config.LOSS.ADJUST_EPOCH)
    adjust_loss_event.attach(trainer)
    trainer.add_event_handler(
        getattr(adjust_loss_event.Events, "EPOCHS_{}_COMPLETED".format(config.LOSS.ADJUST_EPOCH)),
        adjust_loss(criterion, config.LOSS.ADJUSTED_WEIGHTS),
    )

    if silence_other_ranks & rank != 0:
        logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)

    evaluator = create_supervised_evaluator(
        model,
        prepare_val_batch,
        metrics={
            "kaggle": horovod.KaggleMetric(
                output_transform=lambda x: (x["y_pred"], x["mask"])
            ),
            "nll": horovod.LossMetric(
                lovasz_hinge,
                world_size,
                config.TEST.BATCH_SIZE_PER_GPU,
                output_transform=lambda x: (x["y_pred"], x["mask"]),
            ),
        },
        device=device,
        output_transform=padded_val_transform(
            config.TRAIN.PAD_LEFT, config.TRAIN.FINE_SIZE
        ),
    )

    # Set the validation run to start on the epoch completion of the training run
    trainer.add_event_handler(Events.EPOCH_COMPLETED, Evaluator(evaluator, val_loader))

    if rank == 0:  # Run only on master process

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

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED,
            logging_handlers.log_metrics(
                "Validation results",
                metrics_dict={"kaggle": "Kaggle :", "nll": "Avg loss :"},
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
