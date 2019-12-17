# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from ignite.utils import convert_tensor
from scipy.ndimage import zoom
from toolz import compose, curry, itertoolz, pipe


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


def prepare_batch(batch, device=None, non_blocking=False):
    x, y = batch
    return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )


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
        return img_p[:, hdx + ps : hdx + ps + patch_size, wdx + ps : wdx + ps + patch_size]


def compose_processing_pipeline(depth, aug=None):
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
    hdc_wdx_generator = itertools.product(range(0, h - patch_size + ps, stride), range(0, w - patch_size + ps, stride))

    for batch_indexes in itertoolz.partition_all(batch_size, hdc_wdx_generator):
        yield batch_indexes


@curry
def output_processing_pipeline(config, output):
    output = output.unsqueeze(0)
    _, _, h, w = output.shape
    if config.TEST.POST_PROCESSING.SIZE != h or config.TEST.POST_PROCESSING.SIZE != w:
        output = F.interpolate(
            output, size=(config.TEST.POST_PROCESSING.SIZE, config.TEST.POST_PROCESSING.SIZE), mode="bilinear",
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


def patch_label_2d(
    model, img, pre_processing, output_processing, patch_size, stride, batch_size, device, num_classes,
):
    """Processes a whole section"""
    img = torch.squeeze(img)
    h, w = img.shape[-2], img.shape[-1]  # height and width

    # Pad image with patch_size/2:
    ps = int(np.floor(patch_size / 2))  # pad size
    img_p = F.pad(img, pad=(ps, ps, ps, ps), mode="constant", value=0)
    output_p = torch.zeros([1, num_classes, h + 2 * ps, w + 2 * ps])

    # generate output:
    for batch_indexes in _generate_batches(h, w, ps, patch_size, stride, batch_size=batch_size):
        batch = torch.stack(
            [pipe(img_p, _extract_patch(hdx, wdx, ps, patch_size), pre_processing) for hdx, wdx in batch_indexes],
            dim=0,
        )

        model_output = model(batch.to(device))
        for (hdx, wdx), output in zip(batch_indexes, model_output.detach().cpu()):
            output = output_processing(output)
            output_p[:, :, hdx + ps : hdx + ps + patch_size, wdx + ps : wdx + ps + patch_size] += output

    # crop the output_p in the middle
    output = output_p[:, :, ps:-ps, ps:-ps]
    return output


def write_section_file(labels, section_file, config):
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


def plot_aline(aline, labels, xlabel, ylabel="depth"):
    """Plot a section of the data."""
    plt.figure(figsize=(18, 6))
    # data
    plt.subplot(1, 2, 1)
    plt.imshow(aline)
    plt.title("Data")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # mask
    plt.subplot(1, 2, 2)
    plt.imshow(labels)
    plt.xlabel(xlabel)
    plt.title("Label")
