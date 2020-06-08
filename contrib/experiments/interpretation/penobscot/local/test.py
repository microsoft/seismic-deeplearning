# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# To Test:
# python test.py TRAIN.END_EPOCH 1 TRAIN.SNAPSHOTS 1 --cfg "configs/seresnet_unet.yaml" --debug
#
# /* spell-checker: disable */
"""Train models on Penobscot dataset

Test models using PyTorch 

Time to run on single V100: 30 minutes
"""


import logging
import logging.config
from itertools import chain
from os import path

import fire
import numpy as np
import torch
import torchvision
from albumentations import Compose, Normalize, PadIfNeeded, Resize
from ignite.engine import Events
from ignite.metrics import Loss
from ignite.utils import convert_tensor
from toolz import compose, tail, take
from toolz.sandbox.core import unzip
from torch.utils import data

from cv_lib.event_handlers import logging_handlers, tensorboard_handlers
from cv_lib.event_handlers.tensorboard_handlers import create_image_writer, create_summary_writer
from cv_lib.segmentation import models
from cv_lib.segmentation.dutchf3.utils import current_datetime, generate_path, git_branch, git_hash, np_to_tb
from cv_lib.segmentation.metrics import class_accuracy, class_iou, mean_class_accuracy, mean_iou, pixelwise_accuracy
from cv_lib.segmentation.penobscot.engine import create_supervised_evaluator
from cv_lib.utils import load_log_configuration
from deepseismic_interpretation.dutchf3.data import decode_segmap
from deepseismic_interpretation.penobscot.data import get_patch_dataset
from deepseismic_interpretation.penobscot.metrics import InlineMeanIoU
from default import _C as config
from default import update_config


def _prepare_batch(batch, device=None, non_blocking=False):
    x, y, ids, patch_locations = batch
    return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        convert_tensor(y, device=device, non_blocking=non_blocking),
        ids,
        patch_locations,
    )


def _padding_from(config):
    padding_height = config.TEST.AUGMENTATIONS.PAD.HEIGHT - config.TEST.AUGMENTATIONS.RESIZE.HEIGHT
    padding_width = config.TEST.AUGMENTATIONS.PAD.WIDTH - config.TEST.AUGMENTATIONS.RESIZE.WIDTH
    assert padding_height == padding_width, "The padding for the height and width need to be the same"
    return int(padding_height)


def _scale_from(config):
    scale_height = config.TEST.AUGMENTATIONS.PAD.HEIGHT / config.TRAIN.PATCH_SIZE
    scale_width = config.TEST.AUGMENTATIONS.PAD.WIDTH / config.TRAIN.PATCH_SIZE
    assert (
        config.TEST.AUGMENTATIONS.PAD.HEIGHT % config.TRAIN.PATCH_SIZE == 0
    ), "The scaling between the patch height and resized height must be whole number"
    assert (
        config.TEST.AUGMENTATIONS.PAD.WIDTH % config.TRAIN.PATCH_SIZE == 0
    ), "The scaling between the patch width and resized height must be whole number"
    assert scale_height == scale_width, "The scaling for the height and width must be the same"
    return int(scale_height)


def _log_tensor_to_tensorboard(images_tensor, identifier, summary_writer, evaluator):
    image_grid = torchvision.utils.make_grid(images_tensor, normalize=False, scale_each=False, nrow=2)
    summary_writer.add_image(identifier, image_grid, evaluator.state.epoch)


_TOP_K = 2  # Number of best performing inlines to log to tensorboard
_BOTTOM_K = 2  # Number of worst performing inlines to log to tensorboard
mask_value = 255


def run(*options, cfg=None, debug=False):
    """Run testing of model

    Notes:
        Options can be passed in via the options argument and loaded from the cfg file
        Options from default.py will be overridden by options loaded from cfg file
        Options passed in via options argument will override option loaded from cfg file
    
    Args:
        *options (str,int ,optional): Options used to overide what is loaded from the
                                      config. To see what options are available consult
                                      default.py
        cfg (str, optional): Location of config file to load. Defaults to None.
    """

    update_config(config, options=options, config_file=cfg)

    # Start logging
    load_log_configuration(config.LOG_CONFIG)
    logger = logging.getLogger(__name__)
    logger.debug(config.WORKERS)
    torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK

    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(seed=config.SEED)

    # Setup Augmentations
    test_aug = Compose(
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

    PenobscotDataset = get_patch_dataset(config)

    test_set = PenobscotDataset(
        config.DATASET.ROOT,
        config.TRAIN.PATCH_SIZE,
        config.TRAIN.STRIDE,
        split="test",
        transforms=test_aug,
        n_channels=config.MODEL.IN_CHANNELS,
        complete_patches_only=config.TEST.COMPLETE_PATCHES_ONLY,
    )

    logger.info(str(test_set))
    n_classes = test_set.n_classes

    test_loader = data.DataLoader(
        test_set, batch_size=config.VALIDATION.BATCH_SIZE_PER_GPU, num_workers=config.WORKERS,
    )

    model = getattr(models, config.MODEL.NAME).get_seg_model(config)
    logger.info(f"Loading model {config.TEST.MODEL_PATH}")
    model.load_state_dict(torch.load(config.TEST.MODEL_PATH), strict=False)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    model = model.to(device)  # Send to GPU

    try:
        output_dir = generate_path(config.OUTPUT_DIR, git_branch(), git_hash(), config.MODEL.NAME, current_datetime(),)
    except TypeError:
        output_dir = generate_path(config.OUTPUT_DIR, config.MODEL.NAME, current_datetime(),)

    summary_writer = create_summary_writer(log_dir=path.join(output_dir, config.LOG_DIR))

    # weights are inversely proportional to the frequency of the classes in
    # the training set
    class_weights = torch.tensor(config.DATASET.CLASS_WEIGHTS, device=device, requires_grad=False)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=mask_value, reduction="mean")

    def _select_pred_and_mask(model_out_dict):
        return (model_out_dict["y_pred"].squeeze(), model_out_dict["mask"].squeeze())

    def _select_all(model_out_dict):
        return (
            model_out_dict["y_pred"].squeeze(),
            model_out_dict["mask"].squeeze(),
            model_out_dict["ids"],
            model_out_dict["patch_locations"],
        )

    inline_mean_iou = InlineMeanIoU(
        config.DATASET.INLINE_HEIGHT,
        config.DATASET.INLINE_WIDTH,
        config.TRAIN.PATCH_SIZE,
        n_classes,
        padding=_padding_from(config),
        scale=_scale_from(config),
        output_transform=_select_all,
    )

    evaluator = create_supervised_evaluator(
        model,
        _prepare_batch,
        metrics={
            "nll": Loss(criterion, output_transform=_select_pred_and_mask, device=device),
            "inIoU": inline_mean_iou,
            "pixa": pixelwise_accuracy(n_classes, output_transform=_select_pred_and_mask, device=device),
            "cacc": class_accuracy(n_classes, output_transform=_select_pred_and_mask, device=device),
            "mca": mean_class_accuracy(n_classes, output_transform=_select_pred_and_mask, device=device),
            "ciou": class_iou(n_classes, output_transform=_select_pred_and_mask, device=device),
            "mIoU": mean_iou(n_classes, output_transform=_select_pred_and_mask, device=device),
        },
        device=device,
    )

    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        logging_handlers.log_metrics(
            "Test results",
            metrics_dict={
                "nll": "Avg loss :",
                "mIoU": "Avg IoU :",
                "pixa": "Pixelwise Accuracy :",
                "mca": "Mean Class Accuracy :",
                "inIoU": "Mean Inline IoU :",
            },
        ),
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        tensorboard_handlers.log_metrics(
            summary_writer,
            evaluator,
            "epoch",
            metrics_dict={"mIoU": "Test/IoU", "nll": "Test/Loss", "mca": "Test/MCA", "inIoU": "Test/MeanInlineIoU",},
        ),
    )

    def _select_max(pred_tensor):
        return pred_tensor.max(1)[1]

    def _tensor_to_numpy(pred_tensor):
        return pred_tensor.squeeze().cpu().numpy()

    transform_func = compose(np_to_tb, decode_segmap, _tensor_to_numpy,)

    transform_pred = compose(transform_func, _select_max)

    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED, create_image_writer(summary_writer, "Test/Image", "image"),
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED, create_image_writer(summary_writer, "Test/Mask", "mask", transform_func=transform_func),
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        create_image_writer(summary_writer, "Test/Pred", "y_pred", transform_func=transform_pred),
    )

    logger.info("Starting training")
    if debug:
        evaluator.run(test_loader, max_epochs=1, epoch_length=1)
    else:
        evaluator.run(test_loader, max_epochs=1, epoch_length=len(test_loader))

    # Log top N and bottom N inlines in terms of IoU to tensorboard
    inline_ious = inline_mean_iou.iou_per_inline()
    sorted_ious = sorted(inline_ious.items(), key=lambda x: x[1], reverse=True)
    topk = ((inline_mean_iou.predictions[key], inline_mean_iou.masks[key]) for key, iou in take(_TOP_K, sorted_ious))
    bottomk = (
        (inline_mean_iou.predictions[key], inline_mean_iou.masks[key]) for key, iou in tail(_BOTTOM_K, sorted_ious)
    )
    stack_and_decode = compose(transform_func, torch.stack)
    predictions, masks = unzip(chain(topk, bottomk))
    predictions_tensor = stack_and_decode(list(predictions))
    masks_tensor = stack_and_decode(list(masks))
    _log_tensor_to_tensorboard(predictions_tensor, "Test/InlinePredictions", summary_writer, evaluator)
    _log_tensor_to_tensorboard(masks_tensor, "Test/InlineMasks", summary_writer, evaluator)

    summary_writer.close()


if __name__ == "__main__":
    fire.Fire(run)
