# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# commitHash: c76bf579a0d5090ebd32426907d051d499f3e847
# url: https://github.com/yalaudah/facies_classification_benchmark

"""
Modified version of the Alaudah testing script
# TODO: Needs to be improved. Needs to be able to run across multiple GPUs and better
#       factoring around the loader
# issue: https://github.com/microsoft/seismic-deeplearning/issues/268
"""

import logging
import logging.config
import os
from os import path

import fire
import numpy as np
import torch
from albumentations import Compose, Normalize
from cv_lib.utils import load_log_configuration
from cv_lib.segmentation import models

from deepseismic_interpretation.dutchf3.data import get_test_loader
from default import _C as config
from default import update_config
from torch.utils import data
from toolz import take


_CLASS_NAMES = [
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
        hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2,).reshape(
            n_class, n_class
        )
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

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
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()  # fraction of the pixels that come from each class
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


def _evaluate_split(split, section_aug, model, device, running_metrics_overall, config, debug=False):
    logger = logging.getLogger(__name__)

    TestSectionLoader = get_test_loader(config)
    test_set = TestSectionLoader(
        data_dir=config.DATASET.ROOT, split=split, is_transform=True, augmentations=section_aug,
    )

    n_classes = test_set.n_classes

    test_loader = data.DataLoader(test_set, batch_size=1, num_workers=config.WORKERS, shuffle=False)
    if debug:
        logger.info("Running in Debug/Test mode")
        test_loader = take(1, test_loader)

    running_metrics_split = runningScore(n_classes)

    # testing mode:
    with torch.no_grad():  # operations inside don't track history
        model.eval()
        total_iteration = 0
        for i, (images, labels) in enumerate(test_loader):
            logger.info(f"split: {split}, section: {i}")
            total_iteration = total_iteration + 1

            outputs = model(images.to(device))

            pred = outputs.detach().max(1)[1].cpu().numpy()
            gt = labels.numpy()
            running_metrics_split.update(gt, pred)
            running_metrics_overall.update(gt, pred)

    # get scores
    score, class_iou = running_metrics_split.get_scores()

    # Log split results
    logger.info(f'Pixel Acc: {score["Pixel Acc: "]:.3f}')
    for cdx, class_name in enumerate(_CLASS_NAMES):
        logger.info(f'  {class_name}_accuracy {score["Class Accuracy: "][cdx]:.3f}')

    logger.info(f'Mean Class Acc: {score["Mean Class Acc: "]:.3f}')
    logger.info(f'Freq Weighted IoU: {score["Freq Weighted IoU: "]:.3f}')
    logger.info(f'Mean IoU: {score["Mean IoU: "]:0.3f}')
    running_metrics_split.reset()


def _write_section_file(labels, section_file):
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


def test(*options, cfg=None, debug=False):
    update_config(config, options=options, config_file=cfg)
    n_classes = config.DATASET.NUM_CLASSES

    # Start logging
    load_log_configuration(config.LOG_CONFIG)
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir, _ = os.path.split(config.TEST.MODEL_PATH)

    # load model:
    model = getattr(models, config.MODEL.NAME).get_seg_model(config)
    model.load_state_dict(torch.load(config.TEST.MODEL_PATH), strict=False)
    model = model.to(device)  # Send to GPU if available

    running_metrics_overall = runningScore(n_classes)

    # Augmentation
    section_aug = Compose([Normalize(mean=(config.TRAIN.MEAN,), std=(config.TRAIN.STD,), max_pixel_value=1,)])

    splits = ["test1", "test2"] if "Both" in config.TEST.SPLIT else [config.TEST.SPLIT]

    for sdx, split in enumerate(splits):
        labels = np.load(path.join(config.DATASET.ROOT, "test_once", split + "_labels.npy"))
        section_file = path.join(config.DATASET.ROOT, "splits", "section_" + split + ".txt")
        _write_section_file(labels, section_file)
        _evaluate_split(split, section_aug, model, device, running_metrics_overall, config, debug=debug)

    # FINAL TEST RESULTS:
    score, class_iou = running_metrics_overall.get_scores()

    logger.info("--------------- FINAL RESULTS -----------------")
    logger.info(f'Pixel Acc: {score["Pixel Acc: "]:.3f}')
    for cdx, class_name in enumerate(_CLASS_NAMES):
        logger.info(f'     {class_name}_accuracy {score["Class Accuracy: "][cdx]:.3f}')
    logger.info(f'Mean Class Acc: {score["Mean Class Acc: "]:.3f}')
    logger.info(f'Freq Weighted IoU: {score["Freq Weighted IoU: "]:.3f}')
    logger.info(f'Mean IoU: {score["Mean IoU: "]:0.3f}')

    # Save confusion matrix:
    confusion = score["confusion_matrix"]
    np.savetxt(path.join(log_dir, "confusion.csv"), confusion, delimiter=" ")


if __name__ == "__main__":
    fire.Fire(test)
