# Copyright (c) Microsoft Corporation. All rights reserved. 
# Licensed under the MIT License. 
# commitHash: c76bf579a0d5090ebd32426907d051d499f3e847
# url: https://github.com/olivesgatech/facies_classification_benchmark
"""Script to generate train and validation sets for Netherlands F3 dataset
"""
import collections
import itertools
import json
import logging
import logging.config
import math
import os
import warnings
from os import path

import fire
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from toolz import curry
from torch.utils import data


def split_non_overlapping_train_val(
    data_dir, stride, per_val=0.2, loader_type="patch"
):
    # create inline and crossline pacthes for training and validation:
    logger = logging.getLogger(__name__)
    SPLITS = path.join(data_dir, "splits")
    labels_path = path.join(data_dir, "train", "train_labels.npy")

    logger.info('Reading data from {data_dir}')
    logger.info('Loading {labels_path}')
    labels = np.load(labels_path)
    iline, xline, depth = labels.shape
    logger.debug(f"Data shape [iline|xline|depth] {labels.shape}")
    # INLINE PATCHES: ------------------------------------------------
    test_iline = math.floor(iline*0.1)
    test_iline_range = itertools.chain(range(0, test_iline),range(iline-test_iline, iline))
    train_line_range = range(test_iline, iline-test_iline)
    horz_locations = range(0, xline - stride, stride)
    vert_locations = range(0, depth - stride, stride)
    logger.debug("Generating Inline patches")
    logger.debug(horz_locations)
    logger.debug(vert_locations)

    def _i_extract_patches(iline_range, horz_locations, vert_locations):
        for i in iline_range:
            locations = ([j, k] for j in horz_locations for k in vert_locations)
            for j, k in locations:
                yield "i_" + str(i) + "_" + str(j) + "_" + str(k)

    test_i_list = list(_i_extract_patches(test_iline_range, horz_locations, vert_locations))
    train_i_list = list(_i_extract_patches(train_line_range, horz_locations, vert_locations))


    # XLINE PATCHES: ------------------------------------------------
    test_xline = math.floor(xline*0.1)
    test_xline_range = itertools.chain(range(0, test_xline),range(xline-test_xline, xline))
    train_xline_range = range(test_xline, xline-test_xline)

    horz_locations = range(0, iline - stride, stride)
    vert_locations = range(0, depth - stride, stride)

    def _x_extract_patches(xline_range, horz_locations, vert_locations):
        for j in xline_range:
            locations = ([i, k] for i in horz_locations for k in vert_locations)
            for i, k in locations:
                yield "x_" + str(i) + "_" + str(j) + "_" + str(k)
           

    test_x_list = list(_x_extract_patches(test_xline_range, horz_locations, vert_locations))
    train_x_list = list(_x_extract_patches(train_xline_range, horz_locations, vert_locations))
    
    train_list = train_x_list + train_i_list
    test_list = test_x_list + test_i_list

    # write to files to disk:
    # NOTE: This isn't quite right we should calculate the patches again for the whole volume
    file_object = open(path.join(SPLITS, loader_type + "_train_val.txt"), "w")
    file_object.write("\n".join(train_list+test_list))
    file_object.close()
    file_object = open(path.join(SPLITS, loader_type + "_train.txt"), "w")
    file_object.write("\n".join(train_list))
    file_object.close()
    file_object = open(path.join(SPLITS, loader_type + "_val.txt"), "w")
    file_object.write("\n".join(test_list))
    file_object.close()



def split_train_val(
    data_dir, stride, per_val=0.2, loader_type="patch", labels_path=LABELS
):
    warnings.warn("THIS CREATES OVERLAPPING TRAINING AND VALIDATION SETS")
    # create inline and crossline pacthes for training and validation:
    logger = logging.getLogger(__name__)
    SPLITS = path.join(data_dir, "splits")
    labels_path = path.join(data_dir, "train", "train_labels.npy")

    logger.info('Reading data from {data_dir}')
    logger.info('Loading {labels_path}')
    labels = np.load(labels_path)
    iline, xline, depth = labels.shape
    logger.debug(f"Data shape [iline|xline|depth] {labels.shape}")
    
    # INLINE PATCHES: ------------------------------------------------
    i_list = []
    horz_locations = range(0, xline - stride, stride)
    vert_locations = range(0, depth - stride, stride)
    logger.debug("Generating Inline patches")
    logger.debug(horz_locations)
    logger.debug(vert_locations)
    for i in range(iline):
        # for every inline:
        # images are references by top-left corner:
        locations = [[j, k] for j in horz_locations for k in vert_locations]
        patches_list = [
            "i_" + str(i) + "_" + str(j) + "_" + str(k) for j, k in locations
        ]
        i_list.append(patches_list)

    # flatten the list
    i_list = list(itertools.chain(*i_list))

    # XLINE PATCHES: ------------------------------------------------
    x_list = []
    horz_locations = range(0, iline - stride, stride)
    vert_locations = range(0, depth - stride, stride)
    for j in range(xline):
        # for every xline:
        # images are references by top-left corner:
        locations = [[i, k] for i in horz_locations for k in vert_locations]
        patches_list = [
            "x_" + str(i) + "_" + str(j) + "_" + str(k) for i, k in locations
        ]
        x_list.append(patches_list)

    # flatten the list
    x_list = list(itertools.chain(*x_list))

    list_train_val = i_list + x_list

    # create train and test splits:
    list_train, list_val = train_test_split(
        list_train_val, test_size=per_val, shuffle=True
    )

    # write to files to disk:
    file_object = open(path.join(SPLITS, loader_type + "_train_val.txt"), "w")
    file_object.write("\n".join(list_train_val))
    file_object.close()
    file_object = open(path.join(SPLITS, loader_type + "_train.txt"), "w")
    file_object.write("\n".join(list_train))
    file_object.close()
    file_object = open(path.join(SPLITS, loader_type + "_val.txt"), "w")
    file_object.write("\n".join(list_val))
    file_object.close()


def split_non_overlapping(stride, fraction_validation = 0.2, log_config=None):
    """Generate train and validation files of non-overlapping segments
    
    Args:
        stride (int): stride to use when sectioning of the volume
        fraction_validation (float, optional): the fraction of the volume to use for validation. Defaults to 0.2.
        log_config ([type], optional): path to log config. Defaults to None.
    """

    if log_config is not None:
        logging.config.fileConfig(config.LOG_CONFIG) 

    split_non_overlapping_train_val(
        stride, per_val=fraction_validation, loader_type="patch"
    )


def split_overlapping(stride, fraction_validation = 0.2, log_config=None):
    """Generate train and validation files of segments
    This is the original spliting method from https://github.com/olivesgatech/facies_classification_benchmark
    DON'T USE SEE NOTES BELOW
    
    Args:
        stride (int): stride to use when sectioning of the volume
        fraction_validation (float, optional): the fraction of the volume to use for validation. Defaults to 0.2.
        log_config ([type], optional): path to log config. Defaults to None.

    Notes:
        Only kept for reproducibility. It generates overlapping train and val which makes validation results unreliable
    """
    if log_config is not None:
        logging.config.fileConfig(config.LOG_CONFIG) 

    split_train_val(
        stride, per_val=fraction_validation, loader_type="patch"
    )


if __name__=="__main__":
    """Example:
    python prepare_data.py split_non_overlapping /mnt/dutchf3 50 --log-config=logging.conf
    """
    fire.Fire({
      'split_non_overlapping': split_non_overlapping,
      'split_overlapping': split_overlapping,
  })
