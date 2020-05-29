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
import fire
import os
import shutil
from toolz import partition_all
import glob


def _create_directory(dir_path, overwrite=False):
    logger = logging.getLogger("__name__")
    if overwrite:
        logger.info(f"Set to overwrite. Removing {dir_path}")
        shutil.rmtree(dir_path)

    try:
        logger.info(f"Creating {dir_path}")
        os.mkdir(dir_path)
        return dir_path
    except FileExistsError as e:
        logger.warn(
            f"Can't write to {dir_path} as it already exists. Please specify \
                overwrite=true or delete folder"
        )
        raise e


def _copy_files(files_iter, new_dir):
    logger = logging.getLogger("__name__")
    for f in files_iter:
        logger.debug(f"Copying {f} to {new_dir}")
        shutil.copy(f, new_dir)


def _split_train_val_test(partition, val_ratio, test_ratio):
    total_samples = len(partition)
    val_samples = math.floor(val_ratio * total_samples)
    test_samples = math.floor(test_ratio * total_samples)
    train_samples = total_samples - (val_samples + test_samples)
    train_list = partition[:train_samples]
    val_list = partition[train_samples : train_samples + val_samples]
    test_list = partition[train_samples + val_samples : train_samples + val_samples + test_samples]
    return train_list, val_list, test_list


def split_inline(data_dir, val_ratio, test_ratio, overwrite=False, exclude_files=None):
    """Splits the inline data into train, val and test.

    Args:
        data_dir (str): path to directory that holds the data
        val_ratio (float): the ratio of the partition that will be used for validation
        test_ratio (float): the ratio of the partition that they should use for testing
        exclude_files (list[str]): filenames to exclude from dataset, such as ones that contain
            artifacts. Example:['image1.tiff']
    """
    num_partitions = 5
    image_dir = os.path.join(data_dir, "inlines")
    dir_paths = (os.path.join(image_dir, ddir) for ddir in ("train", "val", "test"))
    locations_list = [_create_directory(d, overwrite=overwrite) for d in dir_paths]  # train, val, test

    images_iter = glob.iglob(os.path.join(image_dir, "*.tiff"))

    if exclude_files is not None:
        images_list = list(itertools.filterfalse(lambda x: x in exclude_files, images_iter))
    else:
        images_list = list(images_iter)

    num_elements = math.ceil(len(images_list) / num_partitions)
    for partition in partition_all(num_elements, images_list):  # Partition files into N partitions
        for files_list, dest_dir in zip(_split_train_val_test(partition, val_ratio, test_ratio), locations_list):
            _copy_files(files_list, dest_dir)


if __name__ == "__main__":
    """Example:
    python prepare_data.py split_inline --data-dir=/mnt/penobscot --val-ratio=.1 --test-ratio=.2

    """
    fire.Fire({"split_inline": split_inline})
