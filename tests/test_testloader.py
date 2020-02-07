import tempfile
import numpy as np
from interpretation.deepseismic_interpretation.dutchf3.data import get_test_loader
import pytest
import yacs.config
import os
#from albumentations import Compose, Normalize


# npy files dimensions
IL = 5
XL = 10
D = 8

CONFIG_FILE = "/mnt/c/Repos/seismic-deeplearning/examples/interpretation/notebooks/configs/unet.yaml"
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


def test_TestSectionLoader_test1():
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
        open(txt_path, 'a').close()

        TestSectionLoader = get_test_loader(config)
        test_set = TestSectionLoader(data_dir = data_dir, split = 'test1')

        assert_dimensions(test_set)


def test_TestSectionLoader_test2():
    with tempfile.TemporaryDirectory() as data_dir:
        os.makedirs(os.path.join(data_dir, "test_once"))
        os.makedirs(os.path.join(data_dir, "splits"))

        seimic = np.zeros([IL, XL, D])
        generate_npy_files(os.path.join(data_dir, "test_once", "test2_seismic.npy"), seimic)

        A = np.load(os.path.join(data_dir, "test_once", "test2_seismic.npy"))

        labels = np.ones([IL, XL, D])
        generate_npy_files(os.path.join(data_dir, "test_once", "test2_labels.npy"), labels)

        txt_path = os.path.join(data_dir, "splits", "section_test2.txt")
        open(txt_path, 'a').close()

        TestSectionLoader = get_test_loader(config)
        test_set = TestSectionLoader(data_dir = data_dir, split = 'test2')

        assert_dimensions(test_set)


def test_TestSectionLoader_custom():
    with tempfile.TemporaryDirectory() as data_dir:
        os.makedirs(os.path.join(data_dir, "volume_name"))
        os.makedirs(os.path.join(data_dir, "splits"))

        seimic = np.zeros([IL, XL, D])
        generate_npy_files(os.path.join(data_dir, "volume_name", "seismic.npy"), seimic)

        labels = np.ones([IL, XL, D])
        generate_npy_files(os.path.join(data_dir, "volume_name", "labels.npy"), labels)

        txt_path = os.path.join(data_dir, "splits", "section_volume_name.txt")
        open(txt_path, 'a').close()

        # Augmentation
        # section_aug = Compose(
        #     [Normalize(mean=(config.TRAIN.MEAN,), std=(config.TRAIN.STD,), max_pixel_value=1)]
        # )

        TestSectionLoader = get_test_loader(config)
        test_set = TestSectionLoader(data_dir = data_dir,
                                     split = "volume_name",
                                     is_transform = True,
                                     augmentations = None,
                                     seismic_path = os.path.join(data_dir, "volume_name", "seismic.npy"),
                                     label_path = os.path.join(data_dir, "volume_name", "label.npy"))

        assert_dimensions(test_set)

