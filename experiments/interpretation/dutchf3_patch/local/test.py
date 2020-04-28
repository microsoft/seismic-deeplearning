# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# commitHash: c76bf579a0d5090ebd32426907d051d499f3e847
# url: https://github.com/yalaudah/facies_classification_benchmark
#
# To Test:
# python test.py TRAIN.END_EPOCH 1 TRAIN.SNAPSHOTS 1 --cfg "configs/hrnet.yaml" --debug
#
# /* spell-checker: disable */
"""
Modified version of the Alaudah testing script
Runs only on single GPU
"""

import itertools
import logging
import logging.config
import os
from os import path

import fire
import numpy as np
import torch
import torch.nn.functional as F
from albumentations import Compose, Normalize, PadIfNeeded, Resize
from matplotlib import cm
from PIL import Image
from toolz import compose, curry, itertoolz, pipe, take
from torch.utils import data

from cv_lib.segmentation import models
from cv_lib.segmentation.dutchf3.utils import current_datetime, generate_path, git_branch, git_hash
from cv_lib.utils import load_log_configuration
from deepseismic_interpretation.dutchf3.data import add_patch_depth_channels, get_seismic_labels, get_test_loader
from default import _C as config
from default import update_config

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


def normalize(array):
    """
    Normalizes a segmentation mask array to be in [0,1] range
    """
    min = array.min()
    return (array - min) / (array.max() - min)


def mask_to_disk(mask, fname):
    """
    write segmentation mask to disk using a particular colormap
    """
    Image.fromarray(cm.gist_earth(normalize(mask), bytes=True)).save(fname)


def _transform_CHW_to_HWC(numpy_array):
    return np.moveaxis(numpy_array, 0, -1)


def _transform_HWC_to_CHW(numpy_array):
    return np.moveaxis(numpy_array, -1, 0)


@curry
def _apply_augmentation3D(aug, numpy_array):
    assert len(numpy_array.shape) == 3, "This method only accepts 3D arrays"
    patch = _transform_CHW_to_HWC(numpy_array)
    patch = aug(image=patch)["image"]
    return _transform_HWC_to_CHW(patch)


@curry
def _apply_augmentation2D(aug, numpy_array):
    assert len(numpy_array.shape) == 2, "This method only accepts 2D arrays"
    return aug(image=numpy_array)["image"]


_AUGMENTATION = {3: _apply_augmentation3D, 2: _apply_augmentation2D}


@curry
def _apply_augmentation(aug, image):
    if isinstance(image, torch.Tensor):
        image = image.numpy()

    if aug is not None:
        return _AUGMENTATION[len(image.shape)](aug, image)
    else:
        return image


def _add_depth(image):
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    return add_patch_depth_channels(image)


def _to_torch(image):
    if isinstance(image, torch.Tensor):
        return image
    else:
        return torch.from_numpy(image).to(torch.float32)


def _expand_dims_if_necessary(torch_tensor):
    if len(torch_tensor.shape) == 2:
        return torch_tensor.unsqueeze(dim=0)
    else:
        return torch_tensor


@curry
def _extract_patch(hdx, wdx, ps, patch_size, img_p):
    if len(img_p.shape) == 2:  # 2D
        return img_p[hdx + ps : hdx + ps + patch_size, wdx + ps : wdx + ps + patch_size]
    else:  # 3D
        return img_p[
            :, hdx + ps : hdx + ps + patch_size, wdx + ps : wdx + ps + patch_size,
        ]


def _compose_processing_pipeline(depth, aug=None):
    steps = []
    if aug is not None:
        steps.append(_apply_augmentation(aug))

    if depth == "patch":
        steps.append(_add_depth)

    steps.append(_to_torch)
    steps.append(_expand_dims_if_necessary)
    steps.reverse()
    return compose(*steps)


def _generate_batches(h, w, ps, patch_size, stride, batch_size=64):
    hdc_wdx_generator = itertools.product(range(0, h - patch_size + ps, stride), range(0, w - patch_size + ps, stride),)
    for batch_indexes in itertoolz.partition_all(batch_size, hdc_wdx_generator):
        yield batch_indexes


@curry
def _output_processing_pipeline(config, output):
    output = output.unsqueeze(0)
    _, _, h, w = output.shape
    if config.TEST.POST_PROCESSING.SIZE != h or config.TEST.POST_PROCESSING.SIZE != w:
        output = F.interpolate(
            output, size=(config.TEST.POST_PROCESSING.SIZE, config.TEST.POST_PROCESSING.SIZE,), mode="bilinear",
        )

    if config.TEST.POST_PROCESSING.CROP_PIXELS > 0:
        _, _, h, w = output.shape
        output = output[
            :,
            :,
            config.TEST.POST_PROCESSING.CROP_PIXELS : h - config.TEST.POST_PROCESSING.CROP_PIXELS,
            config.TEST.POST_PROCESSING.CROP_PIXELS : w - config.TEST.POST_PROCESSING.CROP_PIXELS,
        ]
    return output.squeeze()


def _patch_label_2d(
    model, img, pre_processing, output_processing, patch_size, stride, batch_size, device, num_classes,
):
    """Processes a whole section
    """
    img = torch.squeeze(img)
    h, w = img.shape[-2], img.shape[-1]  # height and width

    # Pad image with patch_size/2:
    ps = int(np.floor(patch_size / 2))  # pad size
    img_p = F.pad(img, pad=(ps, ps, ps, ps), mode="constant", value=0)
    output_p = torch.zeros([1, num_classes, h + 2 * ps, w + 2 * ps])

    # generate output:
    for batch_indexes in _generate_batches(h, w, ps, patch_size, stride, batch_size=batch_size):
        batch = torch.stack(
            [pipe(img_p, _extract_patch(hdx, wdx, ps, patch_size), pre_processing,) for hdx, wdx in batch_indexes],
            dim=0,
        )

        model_output = model(batch.to(device))
        for (hdx, wdx), output in zip(batch_indexes, model_output.detach().cpu()):
            output = output_processing(output)
            output_p[:, :, hdx + ps : hdx + ps + patch_size, wdx + ps : wdx + ps + patch_size,] += output

    # crop the output_p in the middle
    output = output_p[:, :, ps:-ps, ps:-ps]
    return output


@curry
def to_image(label_mask, n_classes=6):
    label_colours = get_seismic_labels()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], label_mask.shape[2], 3))
    rgb[:, :, :, 0] = r
    rgb[:, :, :, 1] = g
    rgb[:, :, :, 2] = b
    return rgb


def _evaluate_split(
    split, section_aug, model, pre_processing, output_processing, device, running_metrics_overall, config, debug=False,
):
    logger = logging.getLogger(__name__)

    TestSectionLoader = get_test_loader(config)
    test_set = TestSectionLoader(config.DATASET.ROOT, split=split, is_transform=True, augmentations=section_aug,)

    n_classes = test_set.n_classes

    test_loader = data.DataLoader(test_set, batch_size=1, num_workers=config.WORKERS, shuffle=False)

    if debug:
        logger.info("Running in Debug/Test mode")
        test_loader = take(1, test_loader)

    try:
        output_dir = generate_path(
            config.OUTPUT_DIR + "_test", git_branch(), git_hash(), config.MODEL.NAME, current_datetime(),
        )
    except TypeError:
        output_dir = generate_path(config.OUTPUT_DIR + "_test", config.MODEL.NAME, current_datetime(),)

    running_metrics_split = runningScore(n_classes)

    # testing mode:
    with torch.no_grad():  # operations inside don't track history
        model.eval()
        total_iteration = 0
        for i, (images, labels) in enumerate(test_loader):
            logger.info(f"split: {split}, section: {i}")
            total_iteration = total_iteration + 1

            outputs = _patch_label_2d(
                model,
                images,
                pre_processing,
                output_processing,
                config.TRAIN.PATCH_SIZE,
                config.TEST.TEST_STRIDE,
                config.VALIDATION.BATCH_SIZE_PER_GPU,
                device,
                n_classes,
            )

            pred = outputs.detach().max(1)[1].numpy()
            gt = labels.numpy()
            running_metrics_split.update(gt, pred)
            running_metrics_overall.update(gt, pred)

            #  dump images to disk for review
            mask_to_disk(pred.squeeze(), os.path.join(output_dir, f"{i}_pred.png"))
            mask_to_disk(gt.squeeze(), os.path.join(output_dir, f"{i}_gt.png"))

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
    log_dir, model_name = os.path.split(config.TEST.MODEL_PATH)

    # load model:
    model = getattr(models, config.MODEL.NAME).get_seg_model(config)
    model.load_state_dict(torch.load(config.TEST.MODEL_PATH), strict=False)
    model = model.to(device)  # Send to GPU if available

    running_metrics_overall = runningScore(n_classes)

    # Augmentation
    section_aug = Compose([Normalize(mean=(config.TRAIN.MEAN,), std=(config.TRAIN.STD,), max_pixel_value=1,)])

    # TODO: make sure that this is consistent with how normalization and agumentation for train.py
    # issue: https://github.com/microsoft/seismic-deeplearning/issues/270
    patch_aug = Compose(
        [
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

    pre_processing = _compose_processing_pipeline(config.TRAIN.DEPTH, aug=patch_aug)
    output_processing = _output_processing_pipeline(config)

    splits = ["test1", "test2"] if "Both" in config.TEST.SPLIT else [config.TEST.SPLIT]
    for sdx, split in enumerate(splits):
        labels = np.load(path.join(config.DATASET.ROOT, "test_once", split + "_labels.npy"))
        section_file = path.join(config.DATASET.ROOT, "splits", "section_" + split + ".txt")
        _write_section_file(labels, section_file)
        _evaluate_split(
            split,
            section_aug,
            model,
            pre_processing,
            output_processing,
            device,
            running_metrics_overall,
            config,
            debug=debug,
        )

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
