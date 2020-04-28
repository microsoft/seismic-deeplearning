# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Test the extract functions against a variety of SEGY files and trace_header scenarioes
"""
import math
import os.path as path
import tempfile

import numpy as np
import pandas as pd
import pytest

import scripts.prepare_dutchf3 as prep_dutchf3

# Setup
OUTPUT = None
ILINE = 551
XLINE = 1008
DEPTH = 351
ALINE = np.zeros((ILINE, XLINE, DEPTH))
STRIDE = 50
PATCH = 100
PER_VAL = 0.2
LOG_CONFIG = None


def test_get_aline_range_step_one():

    """check if it includes the step in the range if step = 1
    """
    SECTION_STRIDE = 1

    # Test
    output_iline = prep_dutchf3._get_aline_range(ILINE, PER_VAL, SECTION_STRIDE)
    output_xline = prep_dutchf3._get_aline_range(XLINE, PER_VAL, SECTION_STRIDE)

    assert str(output_iline[0].step) == str(SECTION_STRIDE)
    assert str(output_xline[0].step) == str(SECTION_STRIDE)


def test_get_aline_range_step_zero():

    """check if a ValueError exception is raised when section_stride = 0
    """
    with pytest.raises(ValueError, match="section_stride cannot be zero or a negative number"):
        SECTION_STRIDE = 0

        # Test
        output_iline = prep_dutchf3._get_aline_range(ILINE, PER_VAL, SECTION_STRIDE)
        output_xline = prep_dutchf3._get_aline_range(XLINE, PER_VAL, SECTION_STRIDE)

        assert output_iline
        assert output_xline


def test_get_aline_range_negative_step():

    """check if a ValueError exception is raised when section_stride = -1
    """
    with pytest.raises(ValueError, match="section_stride cannot be zero or a negative number"):
        SECTION_STRIDE = -1

        # Test
        output_iline = prep_dutchf3._get_aline_range(ILINE, PER_VAL, SECTION_STRIDE)
        output_xline = prep_dutchf3._get_aline_range(XLINE, PER_VAL, SECTION_STRIDE)

        assert output_iline
        assert output_xline


def test_get_aline_range_float_step():

    """check if a ValueError exception is raised when section_stride = 1.1
    """
    with pytest.raises(TypeError, match="'float' object cannot be interpreted as an integer"):
        SECTION_STRIDE = 1.0

        # Test
        output_iline = prep_dutchf3._get_aline_range(ILINE, PER_VAL, SECTION_STRIDE)
        output_xline = prep_dutchf3._get_aline_range(XLINE, PER_VAL, SECTION_STRIDE)

        assert output_iline
        assert output_xline


def test_get_aline_range_single_digit_step():

    """check if it includes the step in the range if 1 < step < 10
    """
    SECTION_STRIDE = 1
    # Test
    output_iline = prep_dutchf3._get_aline_range(ILINE, PER_VAL, SECTION_STRIDE)
    output_xline = prep_dutchf3._get_aline_range(XLINE, PER_VAL, SECTION_STRIDE)

    assert str(output_iline[0].step) == str(SECTION_STRIDE)
    assert str(output_xline[0].step) == str(SECTION_STRIDE)


def test_get_aline_range_double_digit_step():

    """check if it includes the step in the range if step > 10
    """
    SECTION_STRIDE = 17
    # Test
    output_iline = prep_dutchf3._get_aline_range(ILINE, PER_VAL, SECTION_STRIDE)
    output_xline = prep_dutchf3._get_aline_range(XLINE, PER_VAL, SECTION_STRIDE)

    assert str(output_iline[0].step) == str(SECTION_STRIDE)
    assert str(output_xline[0].step) == str(SECTION_STRIDE)


def test_prepare_dutchf3_patch_step_1():

    """check a complete run for the script in case further changes are needed
    """
    # setting a value to SECTION_STRIDE as needed to test the values
    SECTION_STRIDE = 1
    DIRECTION = "inline"

    # use a temp dir that will be discarded at the end of the execution
    with tempfile.TemporaryDirectory() as tmpdirname:

        # saving the file to be used by the script
        label_file = tmpdirname + "/label_file.npy"
        np.save(label_file, ALINE)

        # stting the output directory to be used by the script
        output = tmpdirname + "/split"

        # calling the main function of the script without SECTION_STRIDE, to check default value
        train_list, val_list = prep_dutchf3.split_patch_train_val(
            label_file=label_file,
            section_stride=SECTION_STRIDE,
            patch_stride=STRIDE,
            split_direction=DIRECTION,
            patch_size=PATCH,
            per_val=PER_VAL,
            log_config=LOG_CONFIG,
        )
        prep_dutchf3._write_split_files(output, train_list, val_list, "patch")

        # reading the file and splitting the data
        patch_train = pd.read_csv(output + "/patch_train.txt", header=None, names=["row"])
        patch_train = pd.DataFrame(patch_train.row.str.split("_").tolist(), columns=["dir", "i", "x", "d"])

        # test patch_train and section_stride=2
        x = list(sorted(set(patch_train.x.astype(int))))
        i = list(sorted(set(patch_train.i.astype(int))))

        if DIRECTION == "crossline":
            assert x[1] - x[0] == SECTION_STRIDE
            assert i[1] - i[0] == STRIDE
        elif DIRECTION == "inline":
            assert x[1] - x[0] == STRIDE
            assert i[1] - i[0] == SECTION_STRIDE

        # reading the file and splitting the data
        patch_val = pd.read_csv(output + "/patch_val.txt", header=None, names=["row"])
        patch_val = pd.DataFrame(patch_val.row.str.split("_").tolist(), columns=["dir", "i", "x", "d"])

        # test patch_val and section_stride=2
        x = list(sorted(set(patch_val.x.astype(int))))
        i = list(sorted(set(patch_val.i.astype(int))))

        if DIRECTION == "crossline":
            assert x[1] - x[0] == 1  # SECTION_STRIDE is only used in training.
            assert i[1] - i[0] == STRIDE
        elif DIRECTION == "inline":
            assert x[1] - x[0] == STRIDE
            assert i[1] - i[0] == 1

        # test validation set is, at least, PER_VAL
        PER_VAL_CHK = len(set(patch_train.x)) / (len(set(patch_train.x)) + len(set(patch_val.x))) * 100
        assert round(PER_VAL_CHK, 0) >= int(PER_VAL * 100)
        PER_VAL_CHK = len(set(patch_train.i)) / (len(set(patch_train.i)) + len(set(patch_val.i))) * 100
        assert round(PER_VAL_CHK, 0) >= int(PER_VAL * 100)


def test_prepare_dutchf3_patch_step_2():

    """check a complete run for the script in case further changes are needed
    """
    # setting a value to SECTION_STRIDE as needed to test the values
    SECTION_STRIDE = 2
    DIRECTION = "crossline"

    # use a temp dir that will be discarded at the end of the execution
    with tempfile.TemporaryDirectory() as tmpdirname:

        # saving the file to be used by the script
        label_file = tmpdirname + "/label_file.npy"
        np.save(label_file, ALINE)

        # stting the output directory to be used by the script
        output = tmpdirname + "/split"

        # calling the main function of the script without SECTION_STRIDE, to check default value
        train_list, val_list = prep_dutchf3.split_patch_train_val(
            label_file=label_file,
            section_stride=SECTION_STRIDE,
            patch_stride=STRIDE,
            split_direction=DIRECTION,
            patch_size=PATCH,
            per_val=PER_VAL,
            log_config=LOG_CONFIG,
        )
        prep_dutchf3._write_split_files(output, train_list, val_list, "patch")

        # reading the file and splitting the data
        patch_train = pd.read_csv(output + "/patch_train.txt", header=None, names=["row"])
        patch_train = pd.DataFrame(patch_train.row.str.split("_").tolist(), columns=["dir", "i", "x", "d"])

        # test patch_train and section_stride=2
        x = list(sorted(set(patch_train.x.astype(int))))
        i = list(sorted(set(patch_train.i.astype(int))))

        if DIRECTION == "crossline":
            assert x[1] - x[0] == SECTION_STRIDE
            assert i[1] - i[0] == STRIDE
        elif DIRECTION == "inline":
            assert x[1] - x[0] == STRIDE
            assert i[1] - i[0] == SECTION_STRIDE

        # reading the file and splitting the data
        patch_val = pd.read_csv(output + "/patch_val.txt", header=None, names=["row"])
        patch_val = pd.DataFrame(patch_val.row.str.split("_").tolist(), columns=["dir", "i", "x", "d"])

        # test patch_val and section_stride=2
        x = list(sorted(set(patch_val.x.astype(int))))
        i = list(sorted(set(patch_val.i.astype(int))))

        if DIRECTION == "crossline":
            assert x[1] - x[0] == 1  # SECTION_STRIDE is only used in training.
            assert i[1] - i[0] == STRIDE
        elif DIRECTION == "inline":
            assert x[1] - x[0] == STRIDE
            assert i[1] - i[0] == 1

        # test validation set is, at least, PER_VAL
        PER_VAL_CHK = len(set(patch_train.x)) / (len(set(patch_train.x)) + len(set(patch_val.x))) * 100
        assert round(PER_VAL_CHK, 0) >= int(PER_VAL * 100)
        PER_VAL_CHK = len(set(patch_train.i)) / (len(set(patch_train.i)) + len(set(patch_val.i))) * 100
        assert round(PER_VAL_CHK, 0) >= int(PER_VAL * 100)


def test_prepare_dutchf3_patch_train_and_test_sets_inline():

    """check a complete run for the script in case further changes are needed
    """
    # setting a value to SECTION_STRIDE as needed to test the values
    SECTION_STRIDE = 1
    DIRECTION = "inline"

    # use a temp dir that will be discarded at the end of the execution
    with tempfile.TemporaryDirectory() as tmpdirname:

        # saving the file to be used by the script
        label_file = tmpdirname + "/label_file.npy"
        np.save(label_file, ALINE)

        # stting the output directory to be used by the script
        output = tmpdirname + "/split"

        # calling the main function of the script without SECTION_STRIDE, to check default value
        train_list, val_list = prep_dutchf3.split_patch_train_val(
            label_file=label_file,
            section_stride=SECTION_STRIDE,
            patch_stride=STRIDE,
            split_direction=DIRECTION,
            patch_size=PATCH,
            per_val=PER_VAL,
            log_config=LOG_CONFIG,
        )
        prep_dutchf3._write_split_files(output, train_list, val_list, "patch")

        # reading the file and splitting the data
        patch_train = pd.read_csv(output + "/patch_train.txt", header=None, names=["row"])
        patch_train = patch_train.row.tolist()

        # reading the file and splitting the data
        patch_val = pd.read_csv(output + "/patch_val.txt", header=None, names=["row"])
        patch_val = patch_val.row.tolist()

        # assert patches are unique
        assert set(patch_train) & set(patch_val) == set()

        # test validation set is, at least, PER_VAL
        PER_VAL_CHK = 100 * len(patch_train) / (len(patch_train) + len(patch_val))
        assert round(PER_VAL_CHK, 0) >= int(PER_VAL * 100)
        PER_VAL_CHK = 100 * len(patch_train) / (len(patch_train) + len(patch_val))
        assert round(PER_VAL_CHK, 0) >= int(PER_VAL * 100)


def test_prepare_dutchf3_patch_train_and_test_sets_crossline():

    """check a complete run for the script in case further changes are needed
    """
    # setting a value to SECTION_STRIDE as needed to test the values
    SECTION_STRIDE = 1
    DIRECTION = "crossline"

    # use a temp dir that will be discarded at the end of the execution
    with tempfile.TemporaryDirectory() as tmpdirname:

        # saving the file to be used by the script
        label_file = tmpdirname + "/label_file.npy"
        np.save(label_file, ALINE)

        # stting the output directory to be used by the script
        output = tmpdirname + "/split"

        # calling the main function of the script without SECTION_STRIDE, to check default value
        train_list, val_list = prep_dutchf3.split_patch_train_val(
            label_file=label_file,
            section_stride=SECTION_STRIDE,
            patch_stride=STRIDE,
            split_direction=DIRECTION,
            patch_size=PATCH,
            per_val=PER_VAL,
            log_config=LOG_CONFIG,
        )
        prep_dutchf3._write_split_files(output, train_list, val_list, "patch")

        # reading the file and splitting the data
        patch_train = pd.read_csv(output + "/patch_train.txt", header=None, names=["row"])
        patch_train = patch_train.row.tolist()

        # reading the file and splitting the data
        patch_val = pd.read_csv(output + "/patch_val.txt", header=None, names=["row"])
        patch_val = patch_val.row.tolist()

        # assert patches are unique
        assert set(patch_train) & set(patch_val) == set()

        # test validation set is, at least, PER_VAL
        PER_VAL_CHK = 100 * len(patch_train) / (len(patch_train) + len(patch_val))
        assert round(PER_VAL_CHK, 0) >= int(PER_VAL * 100)
        PER_VAL_CHK = 100 * len(patch_train) / (len(patch_train) + len(patch_val))
        assert round(PER_VAL_CHK, 0) >= int(PER_VAL * 100)


def test_prepare_dutchf3_section_step_1_crossline():

    """check a complete run for the script in case further changes are needed
    """
    # setting a value to SECTION_STRIDE as needed to test the values
    SECTION_STRIDE = 2
    DIRECTION = "crossline"

    # use a temp dir that will be discarded at the end of the execution
    with tempfile.TemporaryDirectory() as tmpdirname:

        # saving the file to be used by the script
        label_file = tmpdirname + "/label_file.npy"
        np.save(label_file, ALINE)

        # stting the output directory to be used by the script
        output = tmpdirname + "/split"

        # calling the main function of the script without SECTION_STRIDE, to check default value
        train_list, val_list = prep_dutchf3.split_section_train_val(
            label_file=label_file,
            section_stride=SECTION_STRIDE,
            split_direction=DIRECTION,
            per_val=PER_VAL,
            log_config=LOG_CONFIG,
        )
        prep_dutchf3._write_split_files(output, train_list, val_list, "section")

        # reading the file and splitting the data
        section_train = pd.read_csv(output + "/section_train.txt", header=None, names=["row"])
        section_train = pd.DataFrame(section_train.row.str.split("_").tolist(), columns=["dir", "section"])

        section_val = pd.read_csv(output + "/section_val.txt", header=None, names=["row"])
        section_val = pd.DataFrame(section_val.row.str.split("_").tolist(), columns=["dir", "section"])

        # test
        assert (float(section_train.section[1]) - float(section_train.section[0])) % float(SECTION_STRIDE) == 0.0
        assert (float(section_val.section[1]) - float(section_val.section[0])) % float(SECTION_STRIDE) > 0.0

        # test validation set is, at least, PER_VAL
        PER_VAL_CHK = len(section_val) / (len(section_val) + len(section_train)) * 100
        assert round(PER_VAL_CHK, 0) >= int(PER_VAL * 100)


def test_prepare_dutchf3_section_step_2_inline():

    """check a complete run for the script in case further changes are needed
    """
    # setting a value to SECTION_STRIDE as needed to test the values
    SECTION_STRIDE = 1
    DIRECTION = "inline"

    # use a temp dir that will be discarded at the end of the execution
    with tempfile.TemporaryDirectory() as tmpdirname:

        # saving the file to be used by the script
        label_file = tmpdirname + "/label_file.npy"
        np.save(label_file, ALINE)

        # stting the output directory to be used by the script
        output = tmpdirname + "/split"

        # calling the main function of the script without SECTION_STRIDE, to check default value
        train_list, val_list = prep_dutchf3.split_section_train_val(
            label_file=label_file,
            section_stride=SECTION_STRIDE,
            split_direction=DIRECTION,
            per_val=PER_VAL,
            log_config=LOG_CONFIG,
        )
        prep_dutchf3._write_split_files(output, train_list, val_list, "section")

        # reading the file and splitting the data
        section_train = pd.read_csv(output + "/section_train.txt", header=None, names=["row"])
        section_train = pd.DataFrame(section_train.row.str.split("_").tolist(), columns=["dir", "section"])

        section_val = pd.read_csv(output + "/section_val.txt", header=None, names=["row"])
        section_val = pd.DataFrame(section_val.row.str.split("_").tolist(), columns=["dir", "section"])

        # test
        assert (float(section_train.section[1]) - float(section_train.section[0])) % float(SECTION_STRIDE) == 0.0
        assert (float(section_val.section[1]) - float(section_val.section[0])) % float(SECTION_STRIDE) == 0.0

        # test validation set is, at least, PER_VAL
        PER_VAL_CHK = len(section_val) / (len(section_val) + len(section_train)) * 100
        assert round(PER_VAL_CHK, 0) >= int(PER_VAL * 100)
