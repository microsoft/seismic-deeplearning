# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# commitHash: c76bf579a0d5090ebd32426907d051d499f3e847
# url: https://github.com/olivesgatech/facies_classification_benchmark
"""Script to generate train and validation sets for Netherlands F3 dataset
"""
import itertools
import logging
import logging.config
import math
import warnings
from os import path

import fire
import numpy as np
from sklearn.model_selection import train_test_split


def _get_splits_path(data_dir):
    return path.join(data_dir, "splits")


def _get_labels_path(data_dir):
    return path.join(data_dir, "train", "train_labels.npy")


def _write_split_files(splits_path, train_list, test_list, loader_type):
    file_object = open(path.join(splits_path, loader_type + "_train_val.txt"), "w")
    file_object.write("\n".join(train_list + test_list))
    file_object.close()
    file_object = open(path.join(splits_path, loader_type + "_train.txt"), "w")
    file_object.write("\n".join(train_list))
    file_object.close()
    file_object = open(path.join(splits_path, loader_type + "_val.txt"), "w")
    file_object.write("\n".join(test_list))
    file_object.close()


def _get_aline_range(aline, per_val):
    # Inline sections
    test_aline = math.floor(aline * per_val / 2)
    test_aline_range = itertools.chain(range(0, test_aline), range(aline - test_aline, aline))
    train_aline_range = range(test_aline, aline - test_aline)

    return train_aline_range, test_aline_range


def split_section_train_val(data_dir, per_val=0.2, log_config=None):
    """Generate train and validation files for Netherlands F3 dataset.

    Args:
        data_dir (str): data directory path
        per_val (float, optional): the fraction of the volume to use for validation.
            Defaults to 0.2.
    """

    if log_config is not None:
        logging.config.fileConfig(log_config)

    logger = logging.getLogger(__name__)

    logger.info("Splitting data into sections .... ")
    logger.info(f"Reading data from {data_dir}")

    labels_path = _get_labels_path(data_dir)
    logger.info(f"Loading {labels_path}")
    labels = np.load(labels_path)
    logger.debug(f"Data shape [iline|xline|depth] {labels.shape}")

    iline, xline, _ = labels.shape
    # Inline sections
    train_iline_range, test_iline_range = _get_aline_range(iline, per_val)
    train_i_list = ["i_" + str(i) for i in train_iline_range]
    test_i_list = ["i_" + str(i) for i in test_iline_range]

    # Xline sections
    train_xline_range, test_xline_range = _get_aline_range(xline, per_val)
    train_x_list = ["x_" + str(x) for x in train_xline_range]
    test_x_list = ["x_" + str(x) for x in test_xline_range]

    train_list = train_x_list + train_i_list
    test_list = test_x_list + test_i_list

    # write to files to disk
    splits_path = _get_splits_path(data_dir)
    _write_split_files(splits_path, train_list, test_list, "section")


def split_patch_train_val(data_dir, stride, patch, per_val=0.2, log_config=None):
    """Generate train and validation files for Netherlands F3 dataset.

    Args:
        data_dir (str): data directory path
        stride (int): stride to use when sectioning of the volume
        patch (int): size of patch to extract
        per_val (float, optional): the fraction of the volume to use for validation.
            Defaults to 0.2.
    """

    if log_config is not None:
        logging.config.fileConfig(log_config)

    logger = logging.getLogger(__name__)

    logger.info("Splitting data into patches .... ")
    logger.info(f"Reading data from {data_dir}")

    labels_path = _get_labels_path(data_dir)
    logger.info(f"Loading {labels_path}")
    labels = np.load(labels_path)
    logger.debug(f"Data shape [iline|xline|depth] {labels.shape}")

    iline, xline, depth = labels.shape
    # Inline sections
    train_iline_range, test_iline_range = _get_aline_range(iline, per_val)

    # Xline sections
    train_xline_range, test_xline_range = _get_aline_range(xline, per_val)

    # Generate patches from sections
    # Process inlines
    horz_locations = range(0, xline - patch, stride)
    vert_locations = range(0, depth - patch, stride)
    logger.debug("Generating Inline patches")
    logger.debug(horz_locations)
    logger.debug(vert_locations)

    def _i_extract_patches(iline_range, horz_locations, vert_locations):
        for i in iline_range:
            locations = ([j, k] for j in horz_locations for k in vert_locations)
            for j, k in locations:
                yield "i_" + str(i) + "_" + str(j) + "_" + str(k)

    test_i_list = list(_i_extract_patches(test_iline_range, horz_locations, vert_locations))
    train_i_list = list(_i_extract_patches(train_iline_range, horz_locations, vert_locations))

    # Process crosslines
    horz_locations = range(0, iline - patch, stride)
    vert_locations = range(0, depth - patch, stride)

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
    splits_path = _get_splits_path(data_dir)
    _write_split_files(splits_path, train_list, test_list, "patch")


_LOADER_TYPES = {"section": split_section_train_val, "patch": split_patch_train_val}


def get_split_function(loader_type):
    return _LOADER_TYPES.get(loader_type, split_patch_train_val)


def run_split_func(loader_type, *args, **kwargs):
    split_func = get_split_function(loader_type)
    split_func(*args, **kwargs)


def split_alaudah_et_al_19(data_dir, stride, fraction_validation=0.2, loader_type="patch", log_config=None):
    """Generate train and validation files (with overlap) for Netherlands F3 dataset.
    The original split method from https://github.com/olivesgatech/facies_classification_benchmark
    DON'T USE, SEE NOTES BELOW

    Args:
        data_dir (str): data directory path
        stride (int): stride to use when sectioning of the volume
        fraction_validation (float, optional): the fraction of the volume to use for validation.
            Defaults to 0.2.
        loader_type (str, optional): type of data loader, can be "patch" or "section".
            Defaults to "patch".
        log_config (str, optional): path to log config. Defaults to None.

    Notes:
        Only kept for reproducibility. It generates overlapping train and val which makes
            validation results unreliable.
    """

    if log_config is not None:
        logging.config.fileConfig(log_config)

    warnings.warn("THIS CREATES OVERLAPPING TRAINING AND VALIDATION SETS")

    assert loader_type in [
        "section",
        "patch",
    ], f"Loader type {loader_type} is not valid. \
        Please specify either 'section' or 'patch' for loader_type"

    # create inline and crossline pacthes for training and validation:
    logger = logging.getLogger(__name__)

    logger.info("Reading data from {data_dir}")

    labels_path = _get_labels_path(data_dir)
    logger.info("Loading {labels_path}")
    labels = np.load(labels_path)
    iline, xline, depth = labels.shape
    logger.debug(f"Data shape [iline|xline|depth] {labels.shape}")

    if loader_type == "section":
        i_list = ["i_" + str(i) for i in range(iline)]
        x_list = ["x_" + str(x) for x in range(xline)]
    elif loader_type == "patch":
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
            patches_list = ["i_" + str(i) + "_" + str(j) + "_" + str(k) for j, k in locations]
            i_list.append(patches_list)

        # flatten the list
        i_list = list(itertools.chain(*i_list))

        x_list = []
        horz_locations = range(0, iline - stride, stride)
        vert_locations = range(0, depth - stride, stride)
        for j in range(xline):
            # for every xline:
            # images are references by top-left corner:
            locations = [[i, k] for i in horz_locations for k in vert_locations]
            patches_list = ["x_" + str(i) + "_" + str(j) + "_" + str(k) for i, k in locations]
            x_list.append(patches_list)

        # flatten the list
        x_list = list(itertools.chain(*x_list))

    list_train_val = i_list + x_list

    # create train and test splits:
    train_list, test_list = train_test_split(list_train_val, test_size=fraction_validation, shuffle=True)

    # write to files to disk:
    splits_path = _get_splits_path(data_dir)
    _write_split_files(splits_path, train_list, test_list, loader_type)


# TODO: Try https://github.com/Chilipp/docrep for doscstring reuse
class SplitTrainValCLI(object):
    def section(self, data_dir, per_val=0.2, log_config=None):
        """Generate section based train and validation files for Netherlands F3 dataset.

        Args:
            data_dir (str): data directory path
            per_val (float, optional):  the fraction of the volume to use for validation.
                Defaults to 0.2.
            log_config (str): path to log configurations
        """
        return split_section_train_val(data_dir, per_val=per_val, log_config=log_config)

    def patch(self, data_dir, stride, patch, per_val=0.2, log_config=None):
        """Generate patch based train and validation files for Netherlands F3 dataset.

        Args:
            data_dir (str): data directory path
            stride (int): stride to use when sectioning of the volume
            patch (int): size of patch to extract
            per_val (float, optional):  the fraction of the volume to use for validation.
                Defaults to 0.2.
            log_config (str): path to log configurations
        """
        return split_patch_train_val(data_dir, stride, patch, per_val=per_val, log_config=log_config)


if __name__ == "__main__":
    """Example:
    python prepare_data.py split_train_val section --data-dir=/mnt/dutch
    or
    python prepare_data.py split_train_val patch --data-dir=/mnt/dutch --stride=50 --patch=100

    """
    fire.Fire(
        {"split_train_val": SplitTrainValCLI, "split_alaudah_et_al_19": split_alaudah_et_al_19,}
    )
