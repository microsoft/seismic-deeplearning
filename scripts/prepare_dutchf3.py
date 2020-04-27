# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# commitHash: c76bf579a0d5090ebd32426907d051d499f3e847
# url: https://github.com/yalaudah/facies_classification_benchmark
"""Script to generate train and validation sets for Netherlands F3 dataset
"""
import itertools
import logging
import logging.config
import math
import warnings
from os import path, mkdir

import fire
import numpy as np
from sklearn.model_selection import train_test_split


def _get_splits_path(data_dir):
    return path.join(data_dir, "splits")


def _get_labels_path(data_dir):
    return path.join(data_dir, "train", "train_labels.npy")


def get_split_function(loader_type):
    return _LOADER_TYPES.get(loader_type, split_patch_train_val)


def run_split_func(loader_type, *args, **kwargs):
    split_func = get_split_function(loader_type)
    split_func(*args, **kwargs)


def _write_split_files(splits_path, train_list, val_list, loader_type):
    if not path.isdir(splits_path):
        mkdir(splits_path)
    file_object = open(path.join(splits_path, loader_type + "_train_val.txt"), "w")
    file_object.write("\n".join(train_list + val_list))
    file_object.close()
    file_object = open(path.join(splits_path, loader_type + "_train.txt"), "w")
    file_object.write("\n".join(train_list))
    file_object.close()
    file_object = open(path.join(splits_path, loader_type + "_val.txt"), "w")
    file_object.write("\n".join(val_list))
    file_object.close()


def _get_aline_range(aline, per_val, section_stride=1):
    """
    Args:
        aline (int): number of seismic sections in the inline or 
            crossline directions
        per_val (float):  the fraction of the volume to use for
            validation. Defaults to 0.2.
        section_stride (int): the stride of the sections in the training data.
            If greater than 1, this function will skip (section_stride-1) between each section 
            Defaults to 1, do not skip any section.
    """
    try:
        if section_stride < 1:
            raise ValueError("section_stride cannot be zero or a negative number")

        if per_val < 0 or per_val >= 1:
            raise ValueError("Validation percentage (per_val) should be a number in the range [0,1).")

        val_aline = math.floor(aline * per_val / 2)
        val_range = itertools.chain(range(0, val_aline), range(aline - val_aline, aline))
        train_range = range(val_aline, aline - val_aline, section_stride)

        return train_range, val_range
    except (Exception, ValueError):
        raise


def split_section_train_val(label_file, split_direction, per_val=0.2, log_config=None, section_stride=1):
    """Generate train and validation files for Netherlands F3 dataset.

    Args:
        label_file (str): npy files with labels. Stored in data_dir
        split_direction (str):  Direction in which to split the data into 
            train & val. Use "inline" or "crossline".
        per_val (float, optional):  the fraction of the volume to use for
            validation. Defaults to 0.2.
        log_config (str): path to log configurations
        section_stride (int): the stride of the sections in the training data.
            If greater than 1, this function will skip (section_stride-1) between each section 
            Defaults to 1, do not skip any section.
    """

    if log_config is not None:
        logging.config.fileConfig(log_config)

    logger = logging.getLogger(__name__)
    logger.info("Splitting data into sections .... ")
    logger.info(f"Loading {label_file}")

    labels = np.load(label_file)
    logger.debug(f"Data shape [iline|xline|depth] {labels.shape}")
    iline, xline, _ = labels.shape  # TODO: Must make sure in the future, all new datasets conform to this order.

    logger.info(f"Splitting in {split_direction} direction.. ")
    if split_direction.lower() == "inline":
        num_sections = iline
        index = "i"
    elif split_direction.lower() == "crossline":
        num_sections = xline
        index = "x"
    else:
        raise ValueError(f"Unknown split_direction {split_direction}")

    train_range, val_range = _get_aline_range(num_sections, per_val, section_stride)
    train_list = [f"{index}_" + str(section) for section in train_range]
    val_list = [f"{index}_" + str(section) for section in val_range]

    return train_list, val_list


def split_patch_train_val(
    label_file, patch_stride, patch_size, split_direction, section_stride=1, per_val=0.2, log_config=None,
):
    """Generate train and validation files for Netherlands F3 dataset.

    Args:
        label_file (str): npy files with labels. Stored in data_dir
        patch_stride (int): stride to use when sampling patches
        patch_size (int): size of patch to extract
        split_direction (str):  Direction in which to split the data into 
            train & val. Use "inline" or "crossline".
        section_stride (int): increment to the slices count.
            If section_stride > 1 the function will skip:
                section_stride - 1 sections in the training data.
            Defaults to 1, do not skip any slice.
        per_val (float, optional):  the fraction of the volume to use for
            validation. Defaults to 0.2.
        log_config (str): path to log configurations
    """

    if log_config is not None:
        logging.config.fileConfig(log_config)

    logger = logging.getLogger(__name__)
    logger.info(f"Splitting data into patches along {split_direction} direction .. ")
    logger.info(f"Loading {label_file}")
    labels = np.load(label_file)
    logger.debug(f"Data shape [iline|xline|depth] {labels.shape}")

    iline, xline, depth = labels.shape

    split_direction = split_direction.lower()
    if split_direction == "inline":
        num_sections, section_length = iline, xline
    elif split_direction == "crossline":
        num_sections, section_length = xline, iline
    else:
        raise ValueError(f"Unknown split_direction: {split_direction}")

    train_range, val_range = _get_aline_range(num_sections, per_val, section_stride)
    vert_locations = range(0, depth, patch_stride)
    horz_locations = range(0, section_length, patch_stride)
    logger.debug(vert_locations)
    logger.debug(horz_locations)

    # Process sections:
    def _extract_patches(sections_range, direction, horz_locations, vert_locations):
        locations = itertools.product(sections_range, horz_locations, vert_locations)
        if direction == "inline":
            idx, xdx, ddx = 0, 1, 2
            dir = "i"
        elif direction == "crossline":
            idx, xdx, ddx = 1, 0, 2
            dir = "x"

        for loc in locations:  #      iline               xline                 depth
            yield f"{dir}_" + str(loc[idx]) + "_" + str(loc[xdx]) + "_" + str(loc[ddx])

    # Process sections - train
    logger.debug("Generating patches..")
    train_list = list(_extract_patches(train_range, split_direction, horz_locations, vert_locations))
    val_list = list(_extract_patches(val_range, split_direction, horz_locations, vert_locations))

    logger.debug(train_range)
    logger.debug(val_range)
    logger.debug(train_list)
    logger.debug(val_list)

    return train_list, val_list


def split_alaudah_et_al_19(
    data_dir, patch_stride, patch_size, fraction_validation=0.2, loader_type="patch", log_config=None
):
    """Generate train and validation files (with overlap) for Netherlands F3 dataset.
    The original split method from https://github.com/yalaudah/facies_classification_benchmark
    DON'T USE, SEE NOTES BELOW

    Args:
        data_dir (str): data directory path
        patch_stride (int): stride to use when sampling patches
        patch_size (int): size of patch to extract
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
        horz_locations = range(0, xline - patch_size + 1, patch_stride)
        vert_locations = range(0, depth - patch_size + 1, patch_stride)
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
        horz_locations = range(0, iline - patch_size + 1, patch_stride)
        vert_locations = range(0, depth - patch_size + 1, patch_stride)
        for j in range(xline):
            # for every xline:
            # images are references by top-left corner:
            locations = [[i, k] for i in horz_locations for k in vert_locations]
            patches_list = ["x_" + str(i) + "_" + str(j) + "_" + str(k) for i, k in locations]
            x_list.append(patches_list)

        # flatten the list
        x_list = list(itertools.chain(*x_list))

    list_train_val = i_list + x_list

    # create train and validation splits:
    train_list, val_list = train_test_split(list_train_val, val_size=fraction_validation, shuffle=True)

    # write to files to disk:
    splits_path = _get_splits_path(data_dir)
    _write_split_files(splits_path, train_list, val_list, loader_type)


class SplitTrainValCLI(object):
    def section(
        self,
        data_dir,
        label_file,
        split_direction,
        per_val=0.2,
        log_config="logging.conf",
        output_dir=None,
        section_stride=1,
    ):
        """Generate section based train and validation files for Netherlands F3
        dataset.

        Args:
            data_dir (str): data directory path
            output_dir (str): directory under data_dir to store the split files
            label_file (str): npy files with labels. Stored in data_dir
            split_direction (int):  Direction in which to split the data into 
                train & val. Use "inline" or "crossline", or "both".
            per_val (float, optional):  the fraction of the volume to use for
                validation. Defaults to 0.2.
            log_config (str): path to log configurations
            section_stride (int): the stride of the sections in the training data.
                If greater than 1, this function will skip (section_stride-1) between each section 
                Defaults to 1, do not skip any section.
        """
        if data_dir is not None:
            label_file = path.join(data_dir, label_file)
            output_dir = path.join(data_dir, output_dir)

        if split_direction.lower() == "both":
            train_list_i, val_list_i = split_section_train_val(
                label_file, "inline", per_val, log_config, section_stride
            )
            train_list_x, val_list_x = split_section_train_val(
                label_file, "crossline", per_val, log_config, section_stride
            )
            # concatenate the two lists:
            train_list = train_list_i + train_list_x
            val_list = val_list_i + val_list_x
        elif split_direction.lower() in ["inline", "crossline"]:
            train_list, val_list = split_section_train_val(
                label_file, split_direction, per_val, log_config, section_stride
            )
        else:
            raise ValueError(f"Unknown split_direction: {split_direction}")
        # write to files to disk
        _write_split_files(output_dir, train_list, val_list, "section")

    def patch(
        self,
        label_file,
        stride,
        patch_size,
        split_direction,
        per_val=0.2,
        log_config="logging.conf",
        data_dir=None,
        output_dir=None,
        section_stride=1,
    ):
        """Generate train and validation files for Netherlands F3 dataset.

        Args:
            data_dir (str): data directory path
            output_dir (str): directory under data_dir to store the split files
            label_file (str): npy files with labels. Stored in data_dir
            stride (int): stride to use when sampling patches
            patch_size (int): size of patch to extract
            per_val (float, optional):  the fraction of the volume to use for
                validation. Defaults to 0.2.
            log_config (str): path to log configurations
            split_direction (int):  Direction in which to split the data into 
                train & val. Use "inline" or "crossline", or "both".
            section_stride (int): the stride of the sections in the training data.
                If greater than 1, this function will skip (section_stride-1) between each section 
                Defaults to 1, do not skip any section.
        """
        if data_dir is not None:
            label_file = path.join(data_dir, label_file)
            output_dir = path.join(data_dir, output_dir)

        if split_direction.lower() == "both":
            train_list_i, val_list_i = split_patch_train_val(
                label_file, stride, patch_size, "inline", section_stride, per_val, log_config
            )

            train_list_x, val_list_x = split_patch_train_val(
                label_file, stride, patch_size, "crossline", section_stride, per_val, log_config
            )
            # concatenate the two lists:
            train_list = train_list_i + train_list_x
            val_list = val_list_i + val_list_x
        elif split_direction.lower() in ["inline", "crossline"]:
            train_list, val_list = split_patch_train_val(
                label_file, stride, patch_size, split_direction, section_stride, per_val, log_config
            )
        else:
            raise ValueError(f"Unknown split_direction: {split_direction}")

        # write to files to disk:
        _write_split_files(output_dir, train_list, val_list, "patch")
        print(f"Successfully created the splits files in {output_dir}")


_LOADER_TYPES = {"section": split_section_train_val, "patch": split_patch_train_val}

if __name__ == "__main__":
    """Example:
    python prepare_data.py split_train_val section --data_dir=data \
        --label_file=label_file.npy --output_dir=splits --split_direction=both --section_stride=2
    or
    python prepare_dutchf3.py split_train_val patch --data_dir=data \
        --label_file=label_file.npy --output_dir=splits --stride=50 \
        --patch_size=100 --split_direction=both --section_stride=2
    """
    fire.Fire(
        {"split_train_val": SplitTrainValCLI}
        # commenting the following line as this was not updated with
        # the new parameters names
        # "split_alaudah_et_al_19": split_alaudah_et_al_19}
    )
