# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import torch
from git import Repo
from datetime import datetime
import os


def np_to_tb(array):
    # if 2D :
    if array.ndim == 2:
        # HW => CHW
        array = np.expand_dims(array, axis=0)
        # CHW => NCHW
        array = np.expand_dims(array, axis=0)
    elif array.ndim == 3:
        # HWC => CHW
        array = array.transpose(2, 0, 1)
        # CHW => NCHW
        array = np.expand_dims(array, axis=0)

    array = torch.from_numpy(array)
    return array


def current_datetime():
    return datetime.now().strftime("%b%d_%H%M%S")


def git_branch():
    repo = Repo(search_parent_directories=True)
    return repo.active_branch.name


def git_hash():
    repo = Repo(search_parent_directories=True)
    return repo.active_branch.commit.hexsha

