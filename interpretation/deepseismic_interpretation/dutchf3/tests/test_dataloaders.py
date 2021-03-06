# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Tests for TrainLoader and TestLoader classes when overriding the file names of the seismic and label data.
"""

import tempfile
import numpy as np
from deepseismic_interpretation.dutchf3.data import (
    get_test_loader,
    TrainPatchLoaderWithDepth,
    TrainSectionLoaderWithDepth,
)
import pytest
import yacs.config
import os

# npy files dimensions
IL = 5
XL = 10
D = 8
N_CLASSES = 2

CONFIG_FILE = "./experiments/interpretation/dutchf3_patch/configs/unet.yaml"

with open(CONFIG_FILE, "rt") as f_read:
    config = yacs.config.load_cfg(f_read)


def generate_npy_files(path, data):
    np.save(path, data)


def assert_dimensions(test_section_loader):
    assert test_section_loader.labels.shape[0] == IL
    assert test_section_loader.labels.shape[1] == XL
    assert test_section_loader.labels.shape[2] == D

    # Because add_section_depth_channels method add
    # 2 extra channels to a 1 channel section
    assert test_section_loader.seismic.shape[0] == IL
    assert test_section_loader.seismic.shape[2] == XL
    assert test_section_loader.seismic.shape[3] == D


def test_TestSectionLoader_should_load_data_from_test1_set():
    with open(CONFIG_FILE, "rt") as f_read:
        config = yacs.config.load_cfg(f_read)

    with tempfile.TemporaryDirectory() as data_dir:
        os.makedirs(os.path.join(data_dir, "test_once"))
        os.makedirs(os.path.join(data_dir, "splits"))

        seimic = np.zeros([IL, XL, D])
        generate_npy_files(os.path.join(data_dir, "test_once", "test1_seismic.npy"), seimic)

        labels = np.ones([IL, XL, D])
        generate_npy_files(os.path.join(data_dir, "test_once", "test1_labels.npy"), labels)

        txt_path = os.path.join(data_dir, "splits", "section_test1.txt")
        open(txt_path, "a").close()

        TestSectionLoader = get_test_loader(config)
        config.merge_from_list(["DATASET.ROOT", data_dir])
        test_set = TestSectionLoader(config, split="test1")

        assert_dimensions(test_set)


def test_TestSectionLoader_should_load_data_from_test2_set():
    with tempfile.TemporaryDirectory() as data_dir:
        os.makedirs(os.path.join(data_dir, "test_once"))
        os.makedirs(os.path.join(data_dir, "splits"))

        seimic = np.zeros([IL, XL, D])
        generate_npy_files(os.path.join(data_dir, "test_once", "test2_seismic.npy"), seimic)

        A = np.load(os.path.join(data_dir, "test_once", "test2_seismic.npy"))

        labels = np.ones([IL, XL, D])
        generate_npy_files(os.path.join(data_dir, "test_once", "test2_labels.npy"), labels)

        txt_path = os.path.join(data_dir, "splits", "section_test2.txt")
        open(txt_path, "a").close()

        TestSectionLoader = get_test_loader(config)
        config.merge_from_list(["DATASET.ROOT", data_dir])
        test_set = TestSectionLoader(config, split="test2")

        assert_dimensions(test_set)


def test_TestSectionLoader_should_load_data_from_path_override_data():
    with tempfile.TemporaryDirectory() as data_dir:
        os.makedirs(os.path.join(data_dir, "volume_name"))
        os.makedirs(os.path.join(data_dir, "splits"))

        seimic = np.zeros([IL, XL, D])
        generate_npy_files(os.path.join(data_dir, "volume_name", "seismic.npy"), seimic)

        labels = np.ones([IL, XL, D])
        generate_npy_files(os.path.join(data_dir, "volume_name", "labels.npy"), labels)

        txt_path = os.path.join(data_dir, "splits", "section_volume_name.txt")
        open(txt_path, "a").close()

        TestSectionLoader = get_test_loader(config)
        config.merge_from_list(["DATASET.ROOT", data_dir])
        test_set = TestSectionLoader(
            config,
            split="volume_name",
            is_transform=True,
            augmentations=None,
            seismic_path=os.path.join(data_dir, "volume_name", "seismic.npy"),
            label_path=os.path.join(data_dir, "volume_name", "labels.npy"),
        )

        assert_dimensions(test_set)


def test_TrainPatchLoaderWithDepth_should_fail_on_missing_seismic_file(tmpdir):
    """
    Check for exception when training param is empty
    """
    # Setup
    os.makedirs(os.path.join(tmpdir, "volume_name"))
    os.makedirs(os.path.join(tmpdir, "splits"))

    labels = np.ones([IL, XL, D])
    generate_npy_files(os.path.join(tmpdir, "volume_name", "labels.npy"), labels)

    txt_path = os.path.join(tmpdir, "splits", "patch_volume_name.txt")
    open(txt_path, "a").close()

    config.merge_from_list(["DATASET.ROOT", str(tmpdir)])

    # Test
    with pytest.raises(Exception) as excinfo:

        _ = TrainPatchLoaderWithDepth(
            config,
            split="volume_name",
            is_transform=True,
            augmentations=None,
            seismic_path=os.path.join(tmpdir, "volume_name", "seismic.npy"),
            label_path=os.path.join(tmpdir, "volume_name", "labels.npy"),
        )
    assert "does not exist" in str(excinfo.value)


def test_TrainPatchLoaderWithDepth_should_fail_on_missing_label_file(tmpdir):
    """
    Check for exception when training param is empty
    """
    # Setup
    os.makedirs(os.path.join(tmpdir, "volume_name"))
    os.makedirs(os.path.join(tmpdir, "splits"))

    seimic = np.zeros([IL, XL, D])
    generate_npy_files(os.path.join(tmpdir, "volume_name", "seismic.npy"), seimic)

    txt_path = os.path.join(tmpdir, "splits", "patch_volume_name.txt")
    open(txt_path, "a").close()

    config.merge_from_list(["DATASET.ROOT", str(tmpdir)])

    # Test
    with pytest.raises(Exception) as excinfo:

        _ = TrainPatchLoaderWithDepth(
            config,
            split="volume_name",
            is_transform=True,
            augmentations=None,
            seismic_path=os.path.join(tmpdir, "volume_name", "seismic.npy"),
            label_path=os.path.join(tmpdir, "volume_name", "labels.npy"),
        )
    assert "does not exist" in str(excinfo.value)


def test_TrainPatchLoaderWithDepth_should_load_with_one_train_and_label_file(tmpdir):
    """
    Check for successful class instantiation w/ single npy file for train & label
    """
    # Setup
    os.makedirs(os.path.join(tmpdir, "volume_name"))
    os.makedirs(os.path.join(tmpdir, "splits"))

    seimic = np.zeros([IL, XL, D])
    generate_npy_files(os.path.join(tmpdir, "volume_name", "seismic.npy"), seimic)

    labels = np.ones([IL, XL, D])
    generate_npy_files(os.path.join(tmpdir, "volume_name", "labels.npy"), labels)

    txt_dir = os.path.join(tmpdir, "splits")
    txt_path = os.path.join(txt_dir, "patch_volume_name.txt")
    open(txt_path, "a").close()

    config.merge_from_list(["DATASET.ROOT", str(tmpdir)])

    # Test
    train_set = TrainPatchLoaderWithDepth(
        config,
        split="volume_name",
        is_transform=True,
        augmentations=None,
        seismic_path=os.path.join(tmpdir, "volume_name", "seismic.npy"),
        label_path=os.path.join(tmpdir, "volume_name", "labels.npy"),
    )

    assert train_set.labels.shape == (IL, XL, D + 2 * config.TRAIN.PATCH_SIZE)
    assert train_set.seismic.shape == (IL, XL, D + 2 * config.TRAIN.PATCH_SIZE)
