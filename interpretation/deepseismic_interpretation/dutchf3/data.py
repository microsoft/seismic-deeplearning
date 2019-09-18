import collections
import itertools
import json
import logging
import math
import os
import warnings
from os import path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from toolz import curry
from torch.utils import data



def _train_data_for(data_dir):
    return path.join(data_dir, "train", "train_seismic.npy")


def _train_labels_for(data_dir):
    return path.join(data_dir, "train", "train_labels.npy")


def _test1_data_for(data_dir):
    return path.join(data_dir, "test_once", "test1_seismic.npy")


def _test1_labels_for(data_dir):
    return path.join(data_dir, "test_once", "test1_labels.npy")


def _test2_data_for(data_dir):
    return path.join(data_dir, "test_once", "test2_seismic.npy")


def _test2_labels_for(data_dir):
    return path.join(data_dir, "test_once", "test2_labels.npy")


class SectionLoader(data.Dataset):
    def __init__(
        self, data_dir, split="train", is_transform=True, augmentations=None
    ):
        self.split = split
        self.data_dir = data_dir
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 6
        self.sections = collections.defaultdict(list)

    def __len__(self):
        return len(self.sections[self.split])

    def __getitem__(self, index):

        section_name = self.sections[self.split][index]
        direction, number = section_name.split(sep="_")

        if direction == "i":
            im = self.seismic[int(number), :, :]
            lbl = self.labels[int(number), :, :]
        elif direction == "x":
            im = self.seismic[:, int(number), :]
            lbl = self.labels[:, int(number), :]

        im, lbl = _transform_WH_to_HW(im), _transform_WH_to_HW(lbl)

        if self.augmentations is not None:
            augmented_dict = self.augmentations(image=im, mask=lbl)
            im, lbl = augmented_dict["image"], augmented_dict["mask"]

        if self.is_transform:
            im, lbl = self.transform(im, lbl)

        return im, lbl

    def transform(self, img, lbl):
        # to be in the BxCxHxW that PyTorch uses:
        lbl = np.expand_dims(lbl, 0)
        if len(img.shape) == 2:
            img = np.expand_dims(img, 0)
        return torch.from_numpy(img).float(), torch.from_numpy(lbl).long()


class TrainSectionLoader(SectionLoader):
    def __init__(
        self, data_dir, split="train", is_transform=True, augmentations=None
    ):
        super(TrainSectionLoader, self).__init__(
            data_dir,
            split=split,
            is_transform=is_transform,
            augmentations=augmentations,
        )

        self.seismic = np.load(_train_data_for(self.data_dir))
        self.labels = np.load(_train_labels_for(self.data_dir))
        for split in ["train", "val", "train_val"]:
            # reading the file names for 'train', 'val', 'trainval'""
            txt_path = path.join(
                self.data_dir, "splits", "section_" + split + ".txt"
            )
            file_list = tuple(open(txt_path, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            self.sections[split] = file_list


class TrainSectionLoaderWithDepth(TrainSectionLoader):
    def __init__(
        self, data_dir, split="train", is_transform=True, augmentations=None
    ):
        super(TrainSectionLoader, self).__init__(
            data_dir,
            split=split,
            is_transform=is_transform,
            augmentations=augmentations,
        )
        self.seismic = add_section_depth_channels(self.seismic)  # NCWH

    def __getitem__(self, index):

        section_name = self.sections[self.split][index]
        direction, number = section_name.split(sep="_")

        if direction == "i":
            im = self.seismic[int(number), :, :, :]
            lbl = self.labels[int(number), :, :]
        elif direction == "x":
            im = self.seismic[:, :, int(number), :]
            lbl = self.labels[ :, int(number), :]

            im = np.swapaxes(im, 0, 1)  # From WCH to CWH

        im, lbl = _transform_WH_to_HW(im), _transform_WH_to_HW(lbl)

        if self.augmentations is not None:
            im = _transform_CHW_to_HWC(im)
            augmented_dict = self.augmentations(image=im, mask=lbl)
            im, lbl = augmented_dict["image"], augmented_dict["mask"]
            im = _transform_HWC_to_CHW(im)

        if self.is_transform:
            im, lbl = self.transform(im, lbl)

        return im, lbl


class TestSectionLoader(SectionLoader):
    def __init__(
        self, data_dir, split="test1", is_transform=True, augmentations=None
    ):
        super(TestSectionLoader, self).__init__(
            data_dir,
            split=split,
            is_transform=is_transform,
            augmentations=augmentations,
        )

        if "test1" in self.split:
            self.seismic = np.load(_test1_data_for(self.data_dir))
            self.labels = np.load(_test1_labels_for(self.data_dir))
        elif "test2" in self.split:
            self.seismic = np.load(_test2_data_for(self.data_dir))
            self.labels = np.load(_test2_labels_for(self.data_dir))

        # We are in test mode. Only read the given split. The other one might not
        # be available.
        txt_path = path.join(
            self.data_dir, "splits", "section_" + split + ".txt"
        )
        file_list = tuple(open(txt_path, "r"))
        file_list = [id_.rstrip() for id_ in file_list]
        self.sections[split] = file_list


class TestSectionLoaderWithDepth(TestSectionLoader):
    def __init__(
        self, data_dir, split="test1", is_transform=True, augmentations=None
    ):
        super(TestSectionLoaderWithDepth, self).__init__(
            data_dir,
            split=split,
            is_transform=is_transform,
            augmentations=augmentations,
        )

        self.seismic = add_section_depth_channels(self.seismic)  # NCWH

    def __getitem__(self, index):

        section_name = self.sections[self.split][index]
        direction, number = section_name.split(sep="_")

        if direction == "i":
            im = self.seismic[int(number), :, :, :]
            lbl = self.labels[int(number), :, :]
        elif direction == "x":
            im = self.seismic[:, :, int(number), :]
            lbl = self.labels[:, int(number), :]

            im = np.swapaxes(im, 0, 1)  # From WCH to CWH

        im, lbl = _transform_WH_to_HW(im), _transform_WH_to_HW(lbl)

        if self.augmentations is not None:
            im = _transform_CHW_to_HWC(im)
            augmented_dict = self.augmentations(image=im, mask=lbl)
            im, lbl = augmented_dict["image"], augmented_dict["mask"]
            im = _transform_HWC_to_CHW(im)

        if self.is_transform:
            im, lbl = self.transform(im, lbl)

        return im, lbl


def _transform_WH_to_HW(numpy_array):
    assert len(numpy_array.shape) >= 2, "This method needs at least 2D arrays"
    return np.swapaxes(numpy_array, -2, -1)


class PatchLoader(data.Dataset):
    """
        Data loader for the patch-based deconvnet
    """

    def __init__(
        self,
        data_dir,
        stride=30,
        patch_size=99,
        is_transform=True,
        augmentations=None,
    ):
        self.data_dir = data_dir
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 6
        self.patches = collections.defaultdict(list)
        self.patch_size = patch_size
        self.stride = stride

    def pad_volume(self, volume):
        """
        Only used for train/val!! Not test.
        """
        return np.pad(
            volume,
            pad_width=self.patch_size,
            mode="constant",
            constant_values=255,
        )

    def __len__(self):
        return len(self.patches[self.split])

    def __getitem__(self, index):

        patch_name = self.patches[self.split][index]
        direction, idx, xdx, ddx = patch_name.split(sep="_")

        # Shift offsets the padding that is added in training
        # shift = self.patch_size if "test" not in self.split else 0
        # TODO: Remember we are cancelling the shift since we no longer pad
        shift = 0
        idx, xdx, ddx = int(idx) + shift, int(xdx) + shift, int(ddx) + shift

        if direction == "i":
            im = self.seismic[
                idx, xdx : xdx + self.patch_size, ddx : ddx + self.patch_size
            ]
            lbl = self.labels[
                idx, xdx : xdx + self.patch_size, ddx : ddx + self.patch_size
            ]
        elif direction == "x":
            im = self.seismic[
                idx : idx + self.patch_size, xdx, ddx : ddx + self.patch_size
            ]
            lbl = self.labels[
                idx : idx + self.patch_size, xdx, ddx : ddx + self.patch_size
            ]

        im, lbl = _transform_WH_to_HW(im), _transform_WH_to_HW(lbl)

        if self.augmentations is not None:
            augmented_dict = self.augmentations(image=im, mask=lbl)
            im, lbl = augmented_dict["image"], augmented_dict["mask"]

        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, lbl

    def transform(self, img, lbl):
        # to be in the BxCxHxW that PyTorch uses:
        lbl = np.expand_dims(lbl, 0)
        if len(img.shape) == 2:
            img = np.expand_dims(img, 0)
        return torch.from_numpy(img).float(), torch.from_numpy(lbl).long()


class TestPatchLoader(PatchLoader):
    def __init__(
        self,
        data_dir,
        stride=30,
        patch_size=99,
        is_transform=True,
        augmentations=None,
    ):
        super(TestPatchLoader, self).__init__(
            data_dir,
            stride=stride,
            patch_size=patch_size,
            is_transform=is_transform,
            augmentations=augmentations,
        )
        self.seismic = np.load(_train_data_for(self.data_dir))
        self.labels = np.load(_train_labels_for(self.data_dir))

        # We are in test mode. Only read the given split. The other one might not
        # be available.
        self.split = "test1"  # TODO: Fix this can also be test2
        txt_path = path.join(
            self.data_dir, "splits", "patch_" + self.split + ".txt"
        )
        patch_list = tuple(open(txt_path, "r"))
        self.patches[split] = patch_list


class TrainPatchLoader(PatchLoader):
    def __init__(
        self,
        data_dir,
        split="train",
        stride=30,
        patch_size=99,
        is_transform=True,
        augmentations=None,
    ):
        super(TrainPatchLoader, self).__init__(
            data_dir,
            stride=stride,
            patch_size=patch_size,
            is_transform=is_transform,
            augmentations=augmentations,
        )
        # self.seismic = self.pad_volume(np.load(seismic_path))
        # self.labels = self.pad_volume(np.load(labels_path))
        warnings.warn("This no longer pads the volume")
        self.seismic = np.load(_train_data_for(self.data_dir))
        self.labels = np.load(_train_labels_for(self.data_dir))
        # We are in train/val mode. Most likely the test splits are not saved yet,
        # so don't attempt to load them.
        self.split = split
        for split in ["train", "val", "train_val"]:
            # reading the file names for 'train', 'val', 'trainval'""
            txt_path = path.join(
                self.data_dir, "splits", "patch_" + split + ".txt"
            )
            patch_list = tuple(open(txt_path, "r"))
            self.patches[split] = patch_list


class TrainPatchLoaderWithDepth(TrainPatchLoader):
    def __init__(
        self,
        data_dir,
        split="train",
        stride=30,
        patch_size=99,
        is_transform=True,
        augmentations=None,
    ):
        super(TrainPatchLoaderWithDepth, self).__init__(
            data_dir,
            stride=stride,
            patch_size=patch_size,
            is_transform=is_transform,
            augmentations=augmentations,
        )
        self.seismic = np.load(_train_data_for(self.data_dir))
        self.labels = np.load(_train_labels_for(self.data_dir))
        # We are in train/val mode. Most likely the test splits are not saved yet,
        # so don't attempt to load them.
        self.split = split
        for split in ["train", "val", "train_val"]:
            # reading the file names for 'train', 'val', 'trainval'""
            txt_path = path.join(
                self.data_dir, "splits", "patch_" + split + ".txt"
            )
            patch_list = tuple(open(txt_path, "r"))
            self.patches[split] = patch_list

    def __getitem__(self, index):

        patch_name = self.patches[self.split][index]
        direction, idx, xdx, ddx = patch_name.split(sep="_")

        # Shift offsets the padding that is added in training
        # shift = self.patch_size if "test" not in self.split else 0
        # TODO: Remember we are cancelling the shift since we no longer pad
        shift = 0
        idx, xdx, ddx = int(idx) + shift, int(xdx) + shift, int(ddx) + shift

        if direction == "i":
            im = self.seismic[
                idx, xdx : xdx + self.patch_size, ddx : ddx + self.patch_size
            ]
            lbl = self.labels[
                idx, xdx : xdx + self.patch_size, ddx : ddx + self.patch_size
            ]
        elif direction == "x":
            im = self.seismic[
                idx : idx + self.patch_size, xdx, ddx : ddx + self.patch_size
            ]
            lbl = self.labels[
                idx : idx + self.patch_size, xdx, ddx : ddx + self.patch_size
            ]

        im, lbl = _transform_WH_to_HW(im), _transform_WH_to_HW(lbl)

        # TODO: Add check for rotation augmentations and raise warning if found
        if self.augmentations is not None:
            augmented_dict = self.augmentations(image=im, mask=lbl)
            im, lbl = augmented_dict["image"], augmented_dict["mask"]

        im = add_patch_depth_channels(im)

        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, lbl


def _transform_CHW_to_HWC(numpy_array):
    return np.moveaxis(numpy_array, 0, -1)


def _transform_HWC_to_CHW(numpy_array):
    return np.moveaxis(numpy_array, -1, 0)


class TrainPatchLoaderWithSectionDepth(TrainPatchLoader):
    def __init__(
        self,
        data_dir,
        split="train",
        stride=30,
        patch_size=99,
        is_transform=True,
        augmentations=None,
    ):
        super(TrainPatchLoaderWithSectionDepth, self).__init__(
            data_dir,
            split=split,
            stride=stride,
            patch_size=patch_size,
            is_transform=is_transform,
            augmentations=augmentations,
        )
        self.seismic = add_section_depth_channels(self.seismic)

    def __getitem__(self, index):

        patch_name = self.patches[self.split][index]
        direction, idx, xdx, ddx = patch_name.split(sep="_")

        # Shift offsets the padding that is added in training
        # shift = self.patch_size if "test" not in self.split else 0
        # TODO: Remember we are cancelling the shift since we no longer pad
        shift = 0
        idx, xdx, ddx = int(idx) + shift, int(xdx) + shift, int(ddx) + shift
        if direction == "i":
            im = self.seismic[
                idx,
                :,
                xdx : xdx + self.patch_size,
                ddx : ddx + self.patch_size,
            ]
            lbl = self.labels[
                idx, xdx : xdx + self.patch_size, ddx : ddx + self.patch_size
            ]
        elif direction == "x":
            im = self.seismic[
                idx : idx + self.patch_size,
                :,
                xdx,
                ddx : ddx + self.patch_size,
            ]
            lbl = self.labels[
                idx : idx + self.patch_size, xdx, ddx : ddx + self.patch_size
            ]
            im = np.swapaxes(im, 0, 1)  # From WCH to CWH

        im, lbl = _transform_WH_to_HW(im), _transform_WH_to_HW(lbl)

        if self.augmentations is not None:
            im = _transform_CHW_to_HWC(im)
            augmented_dict = self.augmentations(image=im, mask=lbl)
            im, lbl = augmented_dict["image"], augmented_dict["mask"]
            im = _transform_HWC_to_CHW(im)

        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, lbl


_TRAIN_PATCH_LOADERS = {
    "section": TrainPatchLoaderWithSectionDepth,
    "patch": TrainPatchLoaderWithDepth,
}

_TRAIN_SECTION_LOADERS = {
    "section": TrainSectionLoaderWithDepth
}


def get_patch_loader(cfg):
    assert cfg.TRAIN.DEPTH in ["section", "patch", "none"], f"Depth {cfg.TRAIN.DEPTH} not supported for patch data. \
            Valid values: section, patch, none."
    return _TRAIN_PATCH_LOADERS.get(cfg.TRAIN.DEPTH, TrainPatchLoader)

def get_section_loader(cfg):
    assert cfg.TRAIN.DEPTH in ["section", "none"], f"Depth {cfg.TRAIN.DEPTH} not supported for section data. \
        Valid values: section, none."
    return _TRAIN_SECTION_LOADERS.get(cfg.TRAIN.DEPTH, TrainSectionLoader)


_TEST_LOADERS = {
    "section": TestSectionLoaderWithDepth,
}


def get_test_loader(cfg):
    return _TEST_LOADERS.get(cfg.TRAIN.DEPTH, TestSectionLoader)


def add_patch_depth_channels(image_array):
    """Add 2 extra channels to a 1 channel numpy array
    One channel is a linear sequence from 0 to 1 starting from the top of the image to the bottom
    The second channel is the product of the input channel and the 'depth' channel
    
    Args:
        image_array (np.array): 1D Numpy array
    
    Returns:
        [np.array]: 3D numpy array
    """
    h, w = image_array.shape
    image = np.zeros([3, h, w])
    image[0] = image_array
    for row, const in enumerate(np.linspace(0, 1, h)):
        image[1, row, :] = const
    image[2] = image[0] * image[1]
    return image


def add_section_depth_channels(sections_numpy):
    """Add 2 extra channels to a 1 channel section
    One channel is a linear sequence from 0 to 1 starting from the top of the section to the bottom
    The second channel is the product of the input channel and the 'depth' channel
    
    Args:
        sections_numpy (numpy array): 3D Matrix (NWH)Image tensor
    
    Returns:
        [pytorch tensor]: 3D image tensor
    """
    n, w, h = sections_numpy.shape
    image = np.zeros([3, n, w, h])
    image[0] = sections_numpy
    for row, const in enumerate(np.linspace(0, 1, h)):
        image[1, :, :, row] = const
    image[2] = image[0] * image[1]
    return np.swapaxes(image, 0, 1)


def get_seismic_labels():
    return np.asarray(
        [
            [69, 117, 180],
            [145, 191, 219],
            [224, 243, 248],
            [254, 224, 144],
            [252, 141, 89],
            [215, 48, 39],
        ]
    )


@curry
def decode_segmap(label_mask, n_classes=6):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (N,H,W) array of integer values denoting
            the class label at each spatial location.
    Returns:
        (np.ndarray): the resulting decoded color image (NCHW).
    """
    label_colours = get_seismic_labels()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros(
        (label_mask.shape[0], label_mask.shape[1], label_mask.shape[2], 3)
    )
    rgb[:, :, :, 0] = r / 255.0
    rgb[:, :, :, 1] = g / 255.0
    rgb[:, :, :, 2] = b / 255.0
    return np.transpose(rgb, (0, 3, 1, 2))
