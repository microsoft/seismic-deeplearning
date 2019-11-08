import glob
import itertools
import os
import random
import warnings
from builtins import FileNotFoundError
from collections import defaultdict
from itertools import filterfalse

import numpy as np
import torch
from PIL import Image
from toolz import compose, take, curry
from toolz import pipe
from torchvision.datasets.utils import iterable_to_str, verify_str_arg
from torch.utils.data import Dataset
import math

from toolz import partition_all

_open_to_array = compose(np.array, Image.open)


class DataNotSplitException(Exception):
    pass


@curry
def _pad_right_and_bottom(pad_size, numpy_array, pad_value=255):
    assert (
        len(numpy_array.shape) == 2
    ), f"_pad_right_and_bottom only accepts 2D arrays. Input is {len(numpy_array.shape)}D"
    return np.pad(
        numpy_array, pad_width=[(0, pad_size), (0, pad_size)], constant_values=pad_value
    )


def _get_classes_and_counts(mask_list):
    class_counts_dict = defaultdict(int)
    for mask in mask_list:
        for class_label, class_count in zip(*np.unique(mask, return_counts=True)):
            class_counts_dict[class_label] += class_count
    return list(class_counts_dict.keys()), list(class_counts_dict.values())


def _combine(mask_array):
    """Combine classes 2 and 3. Reduce all classes above 3 by one
    """
    mask_array[np.logical_or(mask_array == 2, mask_array == 3)] = 2
    for i in filter(lambda x: x > 3, np.unique(mask_array)):
        mask_array[mask_array == i] = i - 1
    return mask_array


def _combine_classes(mask_array_list):
    """Combine classes

    Segmentation implementations using this dataset seem to combine 
    classes 2 and 3 so we are doing the same here and then relabeling the rest

    Args:
        mask_array_list (list): list of mask (numpy.Array)
    """
    return [_combine(mask_array.copy()) for mask_array in mask_array_list]


def _replicate_channels(image_array, n_channels):
    new_image_array = np.zeros((n_channels, image_array.shape[0], image_array.shape[1]))
    for i in range(n_channels):
        new_image_array[i] = image_array
    return new_image_array


def _number_patches_in(height_or_width, patch_size, stride, complete_patches_only=True):
    strides_in_hw = (height_or_width - patch_size) / stride
    if complete_patches_only:
        return int(np.floor(strides_in_hw))
    else:
        return int(np.ceil(strides_in_hw))


def _is_2D(numpy_array):
    return len(numpy_array.shape) == 2


def _is_3D(numpy_array):
    return len(numpy_array.shape) == 3


@curry
def _extract_patches(patch_size, stride, complete_patches_only, img_array, mask_array):
    height, width = img_array.shape[-2], img_array.shape[-1]
    num_h_patches = _number_patches_in(
        height, patch_size, stride, complete_patches_only=complete_patches_only
    )
    num_w_patches = _number_patches_in(
        width, patch_size, stride, complete_patches_only=complete_patches_only
    )
    height_iter = range(0, stride * (num_h_patches + 1), stride)
    width_iter = range(0, stride * (num_w_patches + 1), stride)
    patch_locations = list(itertools.product(height_iter, width_iter))

    image_patch_generator = _generate_patches_for(
        img_array, patch_locations, patch_size
    )
    mask_patch_generator = _generate_patches_for(
        mask_array, patch_locations, patch_size
    )
    return image_patch_generator, mask_patch_generator, patch_locations


def _generate_patches_for(numpy_array, patch_locations, patch_size):
    if _is_2D(numpy_array):
        generate = _generate_patches_from_2D
    elif _is_3D(numpy_array):
        generate = _generate_patches_from_3D
    else:
        raise ValueError("Array is not 2D or 3D")
    return generate(numpy_array, patch_locations, patch_size)


def _generate_patches_from_2D(numpy_array, patch_locations, patch_size):
    return (
        numpy_array[h : h + patch_size, w : w + patch_size].copy()
        for h, w in patch_locations
    )


def _generate_patches_from_3D(numpy_array, patch_locations, patch_size):
    return (
        numpy_array[:, h : h + patch_size, w : w + patch_size].copy()
        for h, w in patch_locations
    )


_STATS_FUNCS = {"mean": np.mean, "std": np.std, "max": np.max}


def _transform_CHW_to_HWC(numpy_array):
    return np.moveaxis(numpy_array, 0, -1)


def _transform_HWC_to_CHW(numpy_array):
    return np.moveaxis(numpy_array, -1, 0)


def _rescale(numpy_array):
    """ Rescale the numpy array by 10000. The maximum value achievable is 32737 
    This will bring the values between -n and n
    """
    return numpy_array / 10000


def _split_train_val_test(partition, val_ratio, test_ratio):
    total_samples = len(partition)
    val_samples = math.floor(val_ratio * total_samples)
    test_samples = math.floor(test_ratio * total_samples)
    train_samples = total_samples - (val_samples + test_samples)
    train_list = partition[:train_samples]
    val_list = partition[train_samples : train_samples + val_samples]
    test_list = partition[
        train_samples + val_samples : train_samples + val_samples + test_samples
    ]
    return train_list, val_list, test_list


class InlinePatchDataset(Dataset):
    """Dataset that returns patches from Penobscot dataset

    Notes:
        Loads inlines only and splits into patches
    """

    _repr_indent = 4

    def __init__(
        self,
        data_array,
        mask_array,
        patch_size,
        stride,
        split="train",
        transforms=None,
        max_inlines=None,
        n_channels=1,
        complete_patches_only=True,
        val_ratio=0.1,
        test_ratio=0.2,
    ):
        """Initialise Penobscot Dataset

        Args:
           root (str): root directory to load data from
           patch_size (int): the size of the patch in pixels
           stride (int): the stride applied when extracting patches
           split (str, optional): what split to load, (train, val, test). Defaults to `train`
           transforms (albumentations.augmentations.transforms, optional): albumentation transforms to apply to patches. Defaults to None
           exclude_files (list[str], optional): list of files to exclude. Defaults to None
           max_inlines (int, optional): maximum number of inlines to load. Defaults to None
           n_channels (int, optional): number of channels that the output should contain. Defaults to 3
           complete_patches_only (bool, optional): whether to load incomplete patches that are padded to patch_size. Defaults to True
        """

        super(InlinePatchDataset, self).__init__()
        self._data_array = data_array
        self._slice_mask_array = mask_array
        self._split = split
        self._max_inlines = max_inlines
        self._n_channels = n_channels
        self._complete_patches_only = complete_patches_only
        self._patch_size = patch_size
        self._stride = stride
        self._image_array = []
        self._mask_array = []
        self._ids = []
        self._patch_locations = []

        self.transforms = transforms

        valid_modes = ("train", "test", "val")
        msg = "Unknown value '{}' for argument split. " "Valid values are {{{}}}."
        msg = msg.format(split, iterable_to_str(valid_modes))
        verify_str_arg(split, "split", valid_modes, msg)

        # Set the patch and stride for the patch extractor
        _extract_patches_from = _extract_patches(
            patch_size, stride, self._complete_patches_only
        )
        num_partitions = 5
        indexes = self._data_array.shape[0]
        num_elements = math.ceil(indexes / num_partitions)
        train_indexes_list = []
        test_indexes_list = []
        val_indexes_list = []

        for partition in partition_all(
            num_elements, range(indexes)
        ):  # Partition files into N partitions
            train_indexes, val_indexes, test_indexes = _split_train_val_test(
                partition, val_ratio, test_ratio
            )
            train_indexes_list.extend(train_indexes)
            test_indexes_list.extend(test_indexes)
            val_indexes_list.extend(val_indexes)

        if split == "train":
            indexes = train_indexes_list
        elif split == "val":
            indexes = val_indexes_list
        elif split == "test":
            indexes = test_indexes_list

        # Extract patches
        for index in indexes:
            img_array = self._data_array[index]
            mask_array = self._slice_mask_array[index]
            self._ids.append(index)
            image_generator, mask_generator, patch_locations = _extract_patches_from(
                img_array, mask_array
            )
            self._patch_locations.extend(patch_locations)

            self._image_array.extend(image_generator)

            self._mask_array.extend(mask_generator)

        assert len(self._image_array) == len(
            self._patch_locations
        ), "The shape is not the same"

        assert (
            len(self._patch_locations) % len(self._ids) == 0
        ), "Something is wrong with the patches"

        self._patches_per_image = int(len(self._patch_locations) / len(self._ids))

        self._classes, self._class_counts = _get_classes_and_counts(self._mask_array)

    def __len__(self):
        return len(self._image_array)

    @property
    def n_classes(self):
        return len(self._classes)

    @property
    def class_proportions(self):
        total = np.sum(self._class_counts)
        return [(i, w / total) for i, w in zip(self._classes, self._class_counts)]

    def _add_extra_channels(self, image):
        if self._n_channels > 1:
            image = _replicate_channels(image, self._n_channels)
        return image

    def __getitem__(self, index):
        image, target, ids, patch_locations = (
            self._image_array[index],
            self._mask_array[index],
            self._ids[index // self._patches_per_image],
            self._patch_locations[index],
        )

        image = self._add_extra_channels(image)
        if _is_2D(image):
            image = np.expand_dims(image, 0)

        if self.transforms is not None:
            image = _transform_CHW_to_HWC(image)
            augmented_dict = self.transforms(image=image, mask=target)
            image, target = augmented_dict["image"], augmented_dict["mask"]
            image = _transform_HWC_to_CHW(image)

        target = np.expand_dims(target, 0)

        return (
            torch.from_numpy(image).float(),
            torch.from_numpy(target).long(),
            ids,
            np.array(patch_locations),
        )

    @property
    def statistics(self):
        flat_image_array = np.concatenate([i.flatten() for i in self._image_array])
        stats = {
            stat: statfunc(flat_image_array) for stat, statfunc in _STATS_FUNCS.items()
        }
        return "Mean: {mean} Std: {std} Max: {max}".format(**stats)

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return ["{}{}".format(head, lines[0])] + [
            "{}{}".format(" " * len(head), line) for line in lines[1:]
        ]

    def extra_repr(self):
        lines = [
            "Split: {_split}",
            "Patch size: {_patch_size}",
            "Stride: {_stride}",
            "Max inlines: {_max_inlines}",
            "Num channels: {_n_channels}",
            f"Num classes: {self.n_classes}",
            f"Class proportions: {self.class_proportions}",
            "Complete patches only: {_complete_patches_only}",
            f"Dataset statistics: {self.statistics}",
        ]
        return "\n".join(lines).format(**self.__dict__)


_TRAIN_PATCH_DATASETS = {"none": InlinePatchDataset}


def get_patch_dataset(cfg):
    """ Return the Dataset class for Penobscot

    Args:
        cfg: yacs config

    Returns:
        PenobscotInlinePatchDataset
    """
    assert str(cfg.TRAIN.DEPTH).lower() in [
        "none"
    ], f"Depth {cfg.TRAIN.DEPTH} not supported for patch data. \
            Valid values: section, patch, none."
    return _TRAIN_PATCH_DATASETS.get(cfg.TRAIN.DEPTH, InlinePatchDataset)

