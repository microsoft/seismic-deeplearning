import argparse
import logging
import logging.config
import os
from datetime import datetime
from os import path
from os.path import join as pjoin

import fire
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from ignite.contrib.handlers import (
    ConcatScheduler,
    CosineAnnealingScheduler,
    CustomPeriodicEvent,
    LinearCyclicalScheduler,
)
from ignite.engine import Events
from ignite.metrics import Loss
from ignite.utils import convert_tensor
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from toolz import compose, curry
from torch.utils import data
from tqdm import tqdm

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
from albumentations import (
    Compose,
    HorizontalFlip,
    GaussNoise,
    Normalize,
    Resize,
    PadIfNeeded,
)
from cv_lib.segmentation.dutchf3.data import (
    decode_segmap,
    get_train_loader,
    split_non_overlapping_train_val,
    split_train_val,
    section_loader,
    add_patch_depth_channels
)
from cv_lib.segmentation.dutchf3.engine import (
    create_supervised_evaluator,
    create_supervised_trainer,
)
from cv_lib.segmentation.dutchf3.metrics import (
    MeanIoU,
    PixelwiseAccuracy,
    apex,
)
from cv_lib.segmentation.dutchf3.utils import (
    current_datetime,
    generate_path,
    git_branch,
    git_hash,
    np_to_tb,
)
    get_data_ids,
    get_distributed_data_loaders,
    kfold_split,
)
    create_supervised_evaluator,
    create_supervised_trainer,
)
from default import _C as config
from default import update_config
from toolz.itertoolz import take
import itertools
from toolz import itertoolz
import cv2

class_names = [
    "upper_ns",
    "middle_ns",
    "lower_ns",
    "rijnland_chalk",
    "scruff",
    "zechstein",
]


class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        mean_acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
        mean_iu = np.nanmean(iu)
        freq = (
            hist.sum(axis=1) / hist.sum()
        )  # fraction of the pixels that come from each class
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Pixel Acc: ": acc,
                "Class Accuracy: ": acc_cls,
                "Mean Class Acc: ": mean_acc_cls,
                "Freq Weighted IoU: ": fwavacc,
                "Mean IoU: ": mean_iu,
                "confusion_matrix": self.confusion_matrix,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


@curry
def _extract_patch(img_p, hdx, wdx, ps, patch_size, aug=None):
    patch = img_p[
        hdx + ps : hdx + ps + patch_size, wdx + ps : wdx + ps + patch_size
    ]
    if aug is not None:
        # TODO: Make depth optional from config
        patch = add_patch_depth_channels(aug(image=patch.numpy())['image'])
        return torch.from_numpy(patch).to(torch.float32)
    else:
        return patch.unsqueeze(dim=0)


def _generate_batches(
    img_p, h, w, ps, patch_size, stride, config, batch_size=64
):
    hdc_wdx_generator = itertools.product(
        range(0, h - patch_size + ps, stride),
        range(0, w - patch_size + ps, stride),
    )

    test_aug = Compose(
        [
            Resize(
                config.TRAIN.AUGMENTATIONS.RESIZE.HEIGHT,
                config.TRAIN.AUGMENTATIONS.RESIZE.WIDTH,
                always_apply=True,
            ),
            PadIfNeeded(
                min_height=config.TRAIN.AUGMENTATIONS.PAD.HEIGHT,
                min_width=config.TRAIN.AUGMENTATIONS.PAD.WIDTH,
                border_mode=cv2.BORDER_CONSTANT,
                always_apply=True,
                mask_value=255,
            ),
        ]
    )

    for batch_indexes in itertoolz.partition_all(
        batch_size, hdc_wdx_generator
    ):
        yield batch_indexes, torch.stack(
            [
                _extract_patch(img_p, hdx, wdx, ps, patch_size, test_aug)
                for hdx, wdx in batch_indexes
            ],
            dim=0,
        )


def patch_label_2d(model, img, patch_size, stride, config, device):
    img = torch.squeeze(img)
    h, w = img.shape  # height and width

    # Pad image with patch_size/2:
    ps = int(np.floor(patch_size / 2))  # pad size
    img_p = F.pad(img, pad=(ps, ps, ps, ps), mode="constant", value=0)

    num_classes = 6
    output_p = torch.zeros([1, num_classes, h + 2 * ps, w + 2 * ps])
    # generate output:
    for batch_indexes, batch in _generate_batches(
        img_p, h, w, ps, patch_size, stride, config, batch_size=config.VALIDATION.BATCH_SIZE_PER_GPU
    ):
        model_output = model(batch.to(device))
        for (hdx, wdx), output in zip(
            batch_indexes, model_output.detach().cpu()
        ):
            output=output.unsqueeze(0)
            if config.TRAIN.AUGMENTATIONS.PAD.HEIGHT > 0:
                height_diff = (config.TRAIN.AUGMENTATIONS.PAD.HEIGHT - config.TRAIN.AUGMENTATIONS.RESIZE.HEIGHT)//2
                output = output[
                    :,
                    :,
                    height_diff : output.shape[2]-height_diff,
                    height_diff : output.shape[3]-height_diff,
                ]
            
            if config.TRAIN.AUGMENTATIONS.RESIZE.HEIGHT > 0:
                output=F.interpolate(output, size=(patch_size, patch_size), mode="bilinear")
            
            output_p[
                :,
                :,
                hdx + ps : hdx + ps + patch_size,
                wdx + ps : wdx + ps + patch_size,
            ] += torch.squeeze(output)

    # crop the output_p in the middke
    output = output_p[:, :, ps:-ps, ps:-ps]
    return output


def test(*options, cfg=None):
    update_config(config, options=options, config_file=cfg)
    logging.config.fileConfig(config.LOG_CONFIG)
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir, model_name = os.path.split(config.TEST.MODEL_PATH)

    # load model:
    model = getattr(models, config.MODEL.NAME).get_seg_model(config)
    model.load_state_dict(torch.load(config.TEST.MODEL_PATH), strict=False)
    model = model.to(device)  # Send to GPU if available

    running_metrics_overall = runningScore(6)

    splits = (
        ["test1", "test2"]
        if "Both" in config.TEST.SPLIT
        else [config.TEST.SPLIT]
    )
    for sdx, split in enumerate(splits):
        DATA_ROOT = path.join("/mnt", "alaudah")
        labels = np.load(
            path.join(DATA_ROOT, "test_once", split + "_labels.npy")
        )
        section_file = path.join(
            DATA_ROOT, "splits", "section_" + split + ".txt"
        )
        # define indices of the array
        irange, xrange, depth = labels.shape

        if config.TEST.INLINE:
            i_list = list(range(irange))
            i_list = ["i_" + str(inline) for inline in i_list]
        else:
            i_list = []

        if config.TEST.CROSSLINE:
            x_list = list(range(xrange))
            x_list = ["x_" + str(crossline) for crossline in x_list]
        else:
            x_list = []

        list_test = i_list + x_list

        file_object = open(section_file, "w")
        file_object.write("\n".join(list_test))
        file_object.close()

        test_set = section_loader(
            is_transform=True, split=split, augmentations=None
        )
        n_classes = test_set.n_classes

        test_loader = data.DataLoader(
            test_set, batch_size=1, num_workers=config.WORKERS, shuffle=False
        )

        running_metrics_split = runningScore(n_classes)

        # testing mode:
        with torch.no_grad():  # operations inside don't track history
            model.eval()
            total_iteration = 0
            for i, (images, labels) in enumerate(test_loader):
                logger.info(f"split: {split}, section: {i}")
                total_iteration = total_iteration + 1
                image_original, labels_original = images, labels

                outputs = patch_label_2d(
                    model=model,
                    img=images,
                    patch_size=config.TRAIN.PATCH_SIZE,
                    stride=config.TEST.TEST_STRIDE,
                    config=config,
                    device=device,
                )

                pred = outputs.detach().max(1)[1].numpy()
                gt = labels.numpy()
                running_metrics_split.update(gt, pred)
                running_metrics_overall.update(gt, pred)
                decoded = decode_segmap(pred)

        # get scores and save in writer()
        score, class_iou = running_metrics_split.get_scores()

        # Add split results to TB:
        logger.info(f'Pixel Acc: {score["Pixel Acc: "]:.3f}')
        for cdx, class_name in enumerate(class_names):
            logger.info(
                f'  {class_name}_accuracy {score["Class Accuracy: "][cdx]:.3f}'
            )

        logger.info(f'Mean Class Acc: {score["Mean Class Acc: "]:.3f}')
        logger.info(f'Freq Weighted IoU: {score["Freq Weighted IoU: "]:.3f}')
        logger.info(f'Mean IoU: {score["Mean IoU: "]:0.3f}')
        running_metrics_split.reset()

    # FINAL TEST RESULTS:
    score, class_iou = running_metrics_overall.get_scores()

    logger.info("--------------- FINAL RESULTS -----------------")
    logger.info(f'Pixel Acc: {score["Pixel Acc: "]:.3f}')
    for cdx, class_name in enumerate(class_names):
        logger.info(
            f'     {class_name}_accuracy {score["Class Accuracy: "][cdx]:.3f}'
        )
    logger.info(f'Mean Class Acc: {score["Mean Class Acc: "]:.3f}')
    logger.info(f'Freq Weighted IoU: {score["Freq Weighted IoU: "]:.3f}')
    logger.info(f'Mean IoU: {score["Mean IoU: "]:0.3f}')

    # Save confusion matrix:
    confusion = score["confusion_matrix"]
    np.savetxt(path.join(log_dir, "confusion.csv"), confusion, delimiter=" ")


if __name__ == "__main__":
    fire.Fire(test)
