from copy import deepcopy
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import cv2
import torch
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from ignite.utils import convert_tensor
from sklearn.model_selection import KFold
import torch.nn.functional as F


def get_data_ids(data_root, train_csv="train.csv", depths_csv="depths.csv"):
    train_id = pd.read_csv(os.path.join(data_root, "train.csv"))["id"].values
    depth_id = pd.read_csv(os.path.join(data_root, "depths.csv"))["id"].values
    test_id = np.setdiff1d(depth_id, train_id)
    return train_id,depth_id,test_id


def add_depth_channels(image_tensor):
    _, h, w = image_tensor.size()
    image = torch.zeros([3, h, w])
    image[0] = image_tensor
    for row, const in enumerate(np.linspace(0, 1, h)):
        image[1, row, :] = const
    image[2] = image[0] * image[1]
    return image


def train_aug(image, mask):
    if np.random.rand() < 0.5:
        image, mask = do_horizontal_flip2(image, mask)

    if np.random.rand() < 0.5:
        c = np.random.choice(3)
        if c == 0:
            image, mask = do_random_shift_scale_crop_pad2(image, mask, 0.2)

        if c == 1:
            image, mask = do_horizontal_shear2(
                image, mask, dx=np.random.uniform(-0.07, 0.07)
            )

        if c == 2:
            image, mask = do_shift_scale_rotate2(
                image, mask, dx=0, dy=0, scale=1, angle=np.random.uniform(0, 15)
            )

    if np.random.rand() < 0.5:
        c = np.random.choice(2)
        if c == 0:
            image = do_brightness_shift(image, np.random.uniform(-0.1, +0.1))
        if c == 1:
            image = do_brightness_multiply(image, np.random.uniform(1 - 0.08, 1 + 0.08))

    return image, mask


class SaltDataset(Dataset):
    def __init__(
        self,
        image_list,
        mode,
        mask_list=None,
        is_tta=False,
        is_semi=False,
        fine_size=202,
        pad_left=0,
        pad_right=0,
    ):
        self.imagelist = image_list
        self.mode = mode
        self.masklist = mask_list
        self.is_tta = is_tta
        self.is_semi = is_semi
        self.fine_size = fine_size
        self.pad_left = pad_left
        self.pad_right = pad_right

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, idx):
        image = deepcopy(self.imagelist[idx])

        if self.mode == "train":
            mask = deepcopy(self.masklist[idx])

            image, mask = train_aug(image, mask)
            label = np.where(mask.sum() == 0, 1.0, 0.0).astype(np.float32)

            if self.fine_size != image.shape[0]:
                image, mask = do_resize2(image, mask, self.fine_size, self.fine_size)

            if self.pad_left != 0:
                image, mask = do_center_pad2(image, mask, self.pad_left, self.pad_right)

            image = image.reshape(
                1,
                self.fine_size + self.pad_left + self.pad_right,
                self.fine_size + self.pad_left + self.pad_right,
            )
            mask = mask.reshape(
                1,
                self.fine_size + self.pad_left + self.pad_right,
                self.fine_size + self.pad_left + self.pad_right,
            )
            image, mask = torch.from_numpy(image), torch.from_numpy(mask)
            image = add_depth_channels(image)
            return image, mask, torch.from_numpy(label)

        elif self.mode == "val":
            mask = deepcopy(self.masklist[idx])
            if self.fine_size != image.shape[0]:
                image, mask = do_resize2(image, mask, self.fine_size, self.fine_size)
            if self.pad_left != 0:
                image = do_center_pad(image, self.pad_left, self.pad_right)

            image = image.reshape(
                1,
                self.fine_size + self.pad_left + self.pad_right,
                self.fine_size + self.pad_left + self.pad_right,
            )
            mask = mask.reshape(1, self.fine_size, self.fine_size)

            image, mask = torch.from_numpy(image), torch.from_numpy(mask)
            image = add_depth_channels(image)

            return image, mask

        elif self.mode == "test":
            if self.is_tta:
                image = cv2.flip(image, 1)
            if self.fine_size != image.shape[0]:
                image = cv2.resize(image, dsize=(self.fine_size, self.fine_size))
            if self.pad_left != 0:
                image = do_center_pad(image, self.pad_left, self.pad_right)

            image = image.reshape(
                1,
                self.fine_size + self.pad_left + self.pad_right,
                self.fine_size + self.pad_left + self.pad_right,
            )
            image = torch.from_numpy(image)
            image = add_depth_channels(image)
            return image


def train_image_fetch(images_id, train_root):
    image_train = np.zeros((images_id.shape[0], 101, 101), dtype=np.float32)
    mask_train = np.zeros((images_id.shape[0], 101, 101), dtype=np.float32)

    for idx, image_id in tqdm(enumerate(images_id), total=images_id.shape[0]):
        image_path = os.path.join(train_root, "images", "{}.png".format(image_id))
        mask_path = os.path.join(train_root, "masks", "{}.png".format(image_id))

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        image_train[idx] = image
        mask_train[idx] = mask

    return image_train, mask_train


def test_image_fetch(test_id, test_root):
    image_test = np.zeros((len(test_id), 101, 101), dtype=np.float32)

    for n, image_id in tqdm(enumerate(test_id), total=len(test_id)):
        image_path = os.path.join(test_root, "images", "{}.png".format(image_id))

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        image_test[n] = img

    return image_test


def kfold_split(examples_index_list, n_splits=5, random_state=42, shuffle=True):
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
    for train_index, test_index in kf.split(examples_index_list):
        yield train_index, test_index


def get_data_loaders(
    train_ids, 
    val_ids, 
    train_batch_size, 
    val_batch_size, 
    fine_size, 
    pad_left, 
    pad_right,
    data_root,
    num_workers=4,
    pin_memory=True,
):
    train_root = os.path.join(data_root, "train")
    image_train, mask_train = train_image_fetch(train_ids, train_root)
    train_data = SaltDataset(
        image_train,
        mode="train",
        mask_list=mask_train,
        fine_size=fine_size,
        pad_left=pad_left,
        pad_right=pad_right,
    )

    train_loader = DataLoader(
        train_data,
        sampler=RandomSampler(train_data),
        batch_size=train_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    image_val, mask_val = train_image_fetch(val_ids, train_root)
    val_data = SaltDataset(
        image_val,
        mode="val",
        mask_list=mask_val,
        fine_size=fine_size,
        pad_left=pad_left,
        pad_right=pad_right,
    )

    val_loader = DataLoader(
        val_data,
        shuffle=False,
        batch_size=val_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def get_distributed_data_loaders(
    train_ids,
    val_ids,
    train_batch_size,
    val_batch_size,
    fine_size,
    pad_left,
    pad_right,
    rank,
    size,
    data_root,
    num_workers=4,
    pin_memory=True,
    
):
    train_root = os.path.join(data_root, "train")
    image_train, mask_train = train_image_fetch(train_ids, train_root)
    train_data = SaltDataset(
        image_train,
        mode="train",
        mask_list=mask_train,
        fine_size=fine_size,
        pad_left=pad_left,
        pad_right=pad_right,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_data, num_replicas=size, rank=rank
    )
    train_loader = DataLoader(
        train_data,
        sampler=train_sampler,
        batch_size=train_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    image_val, mask_val = train_image_fetch(val_ids, train_root)
    val_data = SaltDataset(
        image_val,
        mode="val",
        mask_list=mask_val,
        fine_size=fine_size,
        pad_left=pad_left,
        pad_right=pad_right,
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_data, num_replicas=size, rank=rank
    )

    val_loader = DataLoader(
        val_data,
        sampler=val_sampler,
        batch_size=val_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def prepare_train_batch(batch, device=None, non_blocking=False):
    x, y, _ = batch
    return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )


def prepare_val_batch(batch, device=None, non_blocking=False):
    x, y = batch
    return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )
