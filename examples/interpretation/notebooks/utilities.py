# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import itertools
import os
import urllib
import pathlib
import validators
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yacs
from ignite.utils import convert_tensor
from scipy.ndimage import zoom
from toolz import compose, curry, itertoolz, pipe


class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2,
        ).reshape(n_class, n_class)
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
    hdc_wdx_generator = itertools.product(
        range(0, h - patch_size + ps, stride), range(0, w - patch_size + ps, stride)
    )

    for batch_indexes in itertoolz.partition_all(batch_size, hdc_wdx_generator):
        yield batch_indexes


@curry
def output_processing_pipeline(config, output):
    output = output.unsqueeze(0)
    _, _, h, w = output.shape
    if config.TEST.POST_PROCESSING.SIZE != h or config.TEST.POST_PROCESSING.SIZE != w:
        output = F.interpolate(
            output,
            size=(config.TEST.POST_PROCESSING.SIZE, config.TEST.POST_PROCESSING.SIZE),
            mode="bilinear",
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
    model,
    img,
    pre_processing,
    output_processing,
    patch_size,
    stride,
    batch_size,
    device,
    num_classes,
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
            [
                pipe(img_p, _extract_patch(hdx, wdx, ps, patch_size), pre_processing)
                for hdx, wdx in batch_indexes
            ],
            dim=0,
        )

        model_output = model(batch.to(device))
        for (hdx, wdx), output in zip(batch_indexes, model_output.detach().cpu()):
            output = output_processing(output)
            output_p[
                :, :, hdx + ps : hdx + ps + patch_size, wdx + ps : wdx + ps + patch_size
            ] += output

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


def validate_config_paths(config):
    """Checks that all paths in the config file are valid"""
    # TODO: this is currently hardcoded, in the future, its better to have a more generic solution.
    # issue https://github.com/microsoft/seismic-deeplearning/issues/265

    # Make sure DATASET.ROOT directory exist:
    assert os.path.isdir(config.DATASET.ROOT), (
        "The DATASET.ROOT specified in the config file is not a valid directory."
        f" Please make sure this path is correct: {config.DATASET.ROOT}"
    )

    # if a pretrained model path is specified in the config, it should exist:
    if "PRETRAINED" in config.MODEL.keys():
        assert os.path.isfile(config.MODEL.PRETRAINED), (
            "A pretrained model is specified in the config file but does not exist."
            f" Please make sure this path is correct: {config.MODEL.PRETRAINED}"
        )

    # if a test model path is specified in the config, it should exist:
    if "TEST" in config.keys():
        if "MODEL_PATH" in config.TEST.keys():
            assert os.path.isfile(config.TEST.MODEL_PATH), (
                "The TEST.MODEL_PATH specified in the config file does not exist."
                f" Please make sure this path is correct: {config.TEST.MODEL_PATH}"
            )
            # Furthermore, if this is a HRNet model, the pretrained model path should exist if the test model is specified:
            if "hrnet" in config.MODEL.NAME:
                assert os.path.isfile(config.MODEL.PRETRAINED), (
                    "For an HRNet model, you should specify the MODEL.PRETRAINED path"
                    " in the config file if the TEST.MODEL_PATH is also specified."
                )


def download_pretrained_model(config):
    """
    This function reads the config file and downloads model pretrained on the penobscot or dutch
    f3 datasets from the deepseismicsharedstore Azure storage.

    Pre-trained model is specified with MODEL.PRETRAINED parameter:
    - if it's a URL, model is downloaded from the URL
    - if it's a valid file handle, model is loaded from that file
        - otherwise model is loaded from a pre-made URL which this code creates

    Running this code will overwrite the config.MODEL.PRETRAINED parameter value to the downloaded
    pretrained model. The is the model which training is initialized from. 
    If this parameter is blank, we start from a randomly-initialized model.

    DATASET.ROOT parameter specifies the dataset which the model was pre-trained on

    MODEL.DEPTH optional parameter specified whether or not depth information was used in the model
    and what kind of depth augmentation it was.

    We determine the pre-trained model name from these two parameters.

    """

    # this assumes the name of the dataset is preserved in the path -- this is the default behaviour of the code.
    if "dutch" in config.DATASET.ROOT:
        dataset = "dutch"
    elif "penobscot" in config.DATASET.ROOT:
        dataset = "penobscot"
    else:
        raise NameError(
            "Unknown dataset name. Only dutch f3 and penobscot are currently supported."
        )

    if "hrnet" in config.MODEL.NAME:
        model = "hrnet"
    elif "deconvnet" in config.MODEL.NAME:
        model = "deconvnet"
    elif "unet" in config.MODEL.NAME:
        model = "unet"
    else:
        raise NameError(
            "Unknown model name. Only hrnet, deconvnet, and unet are currently supported."
        )

    # check if the user already supplied a URL, otherwise figure out the URL
    if validators.url(config.MODEL.PRETRAINED):
        url = config.MODEL.PRETRAINED
        print(f"Will use user-supplied URL of '{url}'")
    elif os.path.isfile(config.MODEL.PRETRAINED):
        url = None
        print(f"Will use user-supplied file on local disk of '{config.MODEL.PRETRAINED}'")
    else:
        # As more pretrained models are added, add their URLs below:
        if dataset == "penobscot":
            if model == "hrnet":
                # TODO: the code should check if the model uses patches or sections.
                # issue: https://github.com/microsoft/seismic-deeplearning/issues/266
                url = "https://deepseismicsharedstore.blob.core.windows.net/master-public-models/penobscot_hrnet_patch_section_depth.pth"
            else:
                raise NotImplementedError(
                    "We don't store a pretrained model for Dutch F3 for this model combination yet."
                )
            # add other models here ..
        elif dataset == "dutch":
            # add other models here ..
            if model == "hrnet" and config.TRAIN.DEPTH == "section":
                url = "https://deepseismicsharedstore.blob.core.windows.net/master-public-models/dutchf3_hrnet_patch_section_depth.pth"
            elif model == "hrnet" and config.TRAIN.DEPTH == "patch":
                url = "https://deepseismicsharedstore.blob.core.windows.net/master-public-models/dutchf3_hrnet_patch_patch_depth.pth"
            elif (
                model == "deconvnet"
                and "skip" in config.MODEL.NAME
                and config.TRAIN.DEPTH == "none"
            ):
                url = "http://deepseismicsharedstore.blob.core.windows.net/master-public-models/dutchf3_deconvnetskip_patch_no_depth.pth"

            elif (
                model == "deconvnet"
                and "skip" not in config.MODEL.NAME
                and config.TRAIN.DEPTH == "none"
            ):
                url = "http://deepseismicsharedstore.blob.core.windows.net/master-public-models/dutchf3_deconvnet_patch_no_depth.pth"
            elif model == "unet" and config.TRAIN.DEPTH == "section":
                url = "http://deepseismicsharedstore.blob.core.windows.net/master-public-models/dutchf3_seresnetunet_patch_section_depth.pth"
            else:
                raise NotImplementedError(
                    "We don't store a pretrained model for Dutch F3 for this model combination yet."
                )
        else:
            raise NotImplementedError(
                "We don't store a pretrained model for this dataset/model combination yet."
            )

        print(f"Could not find a user-supplied URL, downloading from '{url}'")

    # make sure the model_dir directory is writeable
    model_dir = config.TRAIN.MODEL_DIR

    if not os.path.isdir(os.path.dirname(model_dir)) or not os.access(
        os.path.dirname(model_dir), os.W_OK
    ):
        print(f"Cannot write to TRAIN.MODEL_DIR={config.TRAIN.MODEL_DIR}")
        home = str(pathlib.Path.home())
        model_dir = os.path.join(home, "models")
        print(f"Will write to TRAIN.MODEL_DIR={model_dir}")

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    if url:
        # Download the pretrained model:
        pretrained_model_path = os.path.join(
            model_dir, "pretrained_" + dataset + "_" + model + ".pth"
        )

        # always redownload the model
        print(
            f"Downloading the pretrained model to '{pretrained_model_path}'. This will take a few mintues.. \n"
        )
        urllib.request.urlretrieve(url, pretrained_model_path)
        print("Model successfully downloaded.. \n")
    else:
        # use same model which was on disk anyway - no download needed
        pretrained_model_path = config.MODEL.PRETRAINED

    # Update config MODEL.PRETRAINED
    # TODO: Only HRNet uses a pretrained model currently.
    # issue https://github.com/microsoft/seismic-deeplearning/issues/267
    opts = [
        "MODEL.PRETRAINED",
        pretrained_model_path,
        "TRAIN.MODEL_DIR",
        model_dir,
        "TEST.MODEL_PATH",
        pretrained_model_path,
    ]
    config.merge_from_list(opts)

    return config
