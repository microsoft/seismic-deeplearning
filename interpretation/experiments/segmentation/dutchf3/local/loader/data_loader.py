import os
import collections
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from os import path
from torch.utils import data

DATA_ROOT = path.join("/mnt", "alaudah")
TRAIN_PATH = path.join(DATA_ROOT, "train")
TEST_PATH = path.join(DATA_ROOT, "test_once")
TRAIN_SEISMIC = path.join(TRAIN_PATH, "train_seismic.npy")
TRAIN_LABELS = path.join(TRAIN_PATH, "train_labels.npy")

TEST1_SEISMIC = path.join(TEST_PATH, "test1_seismic.npy")
TEST2_SEISMIC = path.join(TEST_PATH, "test2_seismic.npy")

TEST1_LABELS = path.join(TEST_PATH, "test1_labels.npy")
TEST2_LABELS = path.join(TEST_PATH, "test2_labels.npy")



class PatchLoader(data.Dataset):
    """
        Data loader for the patch-based deconvnet
    """

    def __init__(
        self, stride=30, patch_size=99, is_transform=True, augmentations=None
    ):
        self.root = DATA_ROOT
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 6
        self.mean = 0.000941  # average of the training data
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

        shift = self.patch_size if "test" not in self.split else 0
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

        if self.augmentations is not None:
            im, lbl = self.augmentations(im, lbl)

        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, lbl

    def transform(self, img, lbl):
        img -= self.mean

        # to be in the BxCxHxW that PyTorch uses:
        img, lbl = img.T, lbl.T

        img = np.expand_dims(img, 0)
        lbl = np.expand_dims(lbl, 0)

        img = torch.from_numpy(img)
        img = img.float()
        lbl = torch.from_numpy(lbl)
        lbl = lbl.long()

        return img, lbl



class TestPatchLoader(PatchLoader):
    def __init__(
        self,
        stride=30,
        patch_size=99,
        is_transform=True,
        augmentations=None,
        seismic_path=TEST1_SEISMIC,
        labels_path=TEST1_LABELS,
    ):
        super(TestPatchLoader, self).__init__(
            stride=stride,
            patch_size=patch_size,
            is_transform=is_transform,
            augmentations=augmentations,
        )
        self.seismic = np.load(seismic_path)
        self.labels = np.load(labels_path)
        print(self.seismic.shape)
        print(self.labels.shape)
        # We are in test mode. Only read the given split. The other one might not
        # be available.
        self.split="test1" #TODO: Fix this can also be test2
        txt_path = path.join(DATA_ROOT, "splits", "patch_" + self.split + ".txt")
        patch_list = tuple(open(txt_path, "r"))
        self.patches[split] = patch_list
        


class TrainPatchLoader(PatchLoader):
    def __init__(
        self,
        split="train",
        stride=30,
        patch_size=99,
        is_transform=True,
        augmentations=None,
        seismic_path=TRAIN_SEISMIC,
        labels_path=TRAIN_LABELS,
    ):
        super(TrainPatchLoader, self).__init__(
            stride=stride,
            patch_size=patch_size,
            is_transform=is_transform,
            augmentations=augmentations,
        )
        self.seismic = self.pad_volume(np.load(seismic_path))
        self.labels = self.pad_volume(np.load(labels_path))
        print(self.seismic.shape)
        print(self.labels.shape)
        # We are in train/val mode. Most likely the test splits are not saved yet,
        # so don't attempt to load them.
        self.split=split
        for split in ["train", "val", "train_val"]:
            # reading the file names for 'train', 'val', 'trainval'""
            txt_path = path.join(DATA_ROOT, "splits", "patch_" + split + ".txt")
            patch_list = tuple(open(txt_path, "r"))
            self.patches[split] = patch_list

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


def decode_segmap(label_mask, n_classes, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
            the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
            in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    label_colours = get_seismic_labels()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

