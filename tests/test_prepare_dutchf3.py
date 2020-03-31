# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Test the extract functions against a variety of SEGY files and trace_header scenarioes
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import scripts.prepare_dutchf3 as prep_dutchf3
import math

# Setup
OUTPUT = None
ILINE = 551
XLINE = 1008
DEPTH = 351
ALINE = np.zeros((ILINE, XLINE, DEPTH))
STRIDE = 100
PATCH = 50
PER_VAL = 0.2
LOG_CONFIG = None


def test_get_aline_range_step_one():

    """check if it includes the step in the range if step = 1
    """
    SLICE_STEPS = 1

    # Test
    output_iline = prep_dutchf3._get_aline_range(ILINE, PER_VAL, SLICE_STEPS)
    output_xline = prep_dutchf3._get_aline_range(XLINE, PER_VAL, SLICE_STEPS)

    assert str(output_iline[0].step) == str(SLICE_STEPS)
    assert str(output_xline[0].step) == str(SLICE_STEPS)


def test_get_aline_range_step_zero():

    """check if a ValueError exception is raised when slice_steps = 0
    """
    with pytest.raises(ValueError, match=r'slice_steps cannot be zero or a negative number'):
        SLICE_STEPS = 0

        # Test
        output_iline = prep_dutchf3._get_aline_range(ILINE, PER_VAL, SLICE_STEPS)
        output_xline = prep_dutchf3._get_aline_range(XLINE, PER_VAL, SLICE_STEPS)

        assert output_iline
        assert output_xline


def test_get_aline_range_negative_step():

    """check if a ValueError exception is raised when slice_steps = -1
    """
    with pytest.raises(ValueError, match='slice_steps cannot be zero or a negative number'):
        SLICE_STEPS = -1

        # Test
        output_iline = prep_dutchf3._get_aline_range(ILINE, PER_VAL, SLICE_STEPS)
        output_xline = prep_dutchf3._get_aline_range(XLINE, PER_VAL, SLICE_STEPS)

        assert output_iline
        assert output_xline


def test_get_aline_range_float_step():

    """check if a ValueError exception is raised when slice_steps = 1.1
    """
    with pytest.raises(TypeError, match="'float' object cannot be interpreted as an integer"):
        SLICE_STEPS = 1.

        # Test
        output_iline = prep_dutchf3._get_aline_range(ILINE, PER_VAL, SLICE_STEPS)
        output_xline = prep_dutchf3._get_aline_range(XLINE, PER_VAL, SLICE_STEPS)

        assert output_iline
        assert output_xline


def test_get_aline_range_single_digit_step():

    """check if it includes the step in the range if 1 < step < 10
    """
    SLICE_STEPS = 1
    # Test
    output_iline = prep_dutchf3._get_aline_range(ILINE, PER_VAL, SLICE_STEPS)
    output_xline = prep_dutchf3._get_aline_range(XLINE, PER_VAL, SLICE_STEPS)

    assert str(output_iline[0].step) == str(SLICE_STEPS)
    assert str(output_xline[0].step) == str(SLICE_STEPS)


def test_get_aline_range_double_digit_step():

    """check if it includes the step in the range if step > 10
    """
    SLICE_STEPS = 17
    # Test
    output_iline = prep_dutchf3._get_aline_range(ILINE, PER_VAL, SLICE_STEPS)
    output_xline = prep_dutchf3._get_aline_range(XLINE, PER_VAL, SLICE_STEPS)

    assert str(output_iline[0].step) == str(SLICE_STEPS)
    assert str(output_xline[0].step) == str(SLICE_STEPS)


def test_prepare_dutchf3_patch_step_1():

    """check a complete run for the script in case further changes are needed
    """
    # setting a value to SLICE_STEPS as needed to test the values
    SLICE_STEPS = 1

    # use a temp dir that will be discarded at the end of the execution
    with tempfile.TemporaryDirectory() as tmpdirname:

        # saving the file to be used by the script
        label_file = tmpdirname + '/label_file.npy'
        np.save(label_file, ALINE)

        # stting the output directory to be used by the script
        output = tmpdirname + '/split'

        # calling the main function of the script without SLICE_STEPS, to check default value
        prep_dutchf3.split_patch_train_val(data_dir=tmpdirname, output_dir=output, label_file=label_file, 
        slice_steps=SLICE_STEPS, stride=STRIDE, patch_size=PATCH, per_val=PER_VAL,log_config=LOG_CONFIG)

        # reading the file and splitting the data
        patch_train = pd.read_csv(output + '/patch_train.txt', header=None, names=['row', 'a', 'b'])
        patch_train = pd.DataFrame(patch_train.row.str.split('_').tolist(), columns=['aline', 'x', 'y', 'z'])

        # test patch_train and slice_steps=2
        y = list(sorted(set(patch_train.y.astype(int))))
        x = list(sorted(set(patch_train.x.astype(int))))
        assert (int(y[1]) - int(y[0])) == SLICE_STEPS
        assert (int(x[1]) - int(x[0])) == SLICE_STEPS

        # reading the file and splitting the data
        patch_val = pd.read_csv(output + '/patch_val.txt', header=None, names=['row', 'a', 'b'])
        patch_val = pd.DataFrame(patch_val.row.str.split('_').tolist(), columns=['aline', 'x', 'y', 'z'])

        # test patch_val and slice_steps=2
        y = list(sorted(set(patch_val.y.astype(int))))
        x = list(sorted(set(patch_val.x.astype(int))))
        assert (int(y[1]) - int(y[0])) != SLICE_STEPS
        assert (int(x[1]) - int(x[0])) != SLICE_STEPS

        # test validation set is, at least, PER_VAL
        PER_VAL_CHK = len(set(patch_train.y))/(len(set(patch_train.y))+len(set(patch_val.y))) * 100
        assert round(PER_VAL_CHK,0) >= int(PER_VAL * 100)
        PER_VAL_CHK = len(set(patch_train.x))/(len(set(patch_train.x))+len(set(patch_val.x))) * 100
        assert round(PER_VAL_CHK,0) >= int(PER_VAL * 100)


def test_prepare_dutchf3_patch_step_2():

    """check a complete run for the script in case further changes are needed
    """
    # setting a value to SLICE_STEPS as needed to test the values
    SLICE_STEPS = 2

    # use a temp dir that will be discarded at the end of the execution
    with tempfile.TemporaryDirectory() as tmpdirname:

        # saving the file to be used by the script
        label_file = tmpdirname + '/label_file.npy'
        np.save(label_file, ALINE)

        # stting the output directory to be used by the script
        output = tmpdirname + '/split'

        # calling the main function of the script without SLICE_STEPS, to check default value
        prep_dutchf3.split_patch_train_val(data_dir=tmpdirname, output_dir=output, label_file=label_file, 
        slice_steps=SLICE_STEPS, stride=STRIDE, patch_size=PATCH, per_val=PER_VAL,log_config=LOG_CONFIG)

        # reading the file and splitting the data
        patch_train = pd.read_csv(output + '/patch_train.txt', header=None, names=['row', 'a', 'b'])
        patch_train = pd.DataFrame(patch_train.row.str.split('_').tolist(), columns=['aline', 'x', 'y', 'z'])

        # test patch_train and slice_steps=2
        y = list(sorted(set(patch_train.y.astype(int))))
        x = list(sorted(set(patch_train.x.astype(int))))
        assert (int(y[1]) - int(y[0])) == SLICE_STEPS
        assert (int(x[1]) - int(x[0])) == SLICE_STEPS

        # reading the file and splitting the data
        patch_val = pd.read_csv(output + '/patch_val.txt', header=None, names=['row', 'a', 'b'])
        patch_val = pd.DataFrame(patch_val.row.str.split('_').tolist(), columns=['aline', 'x', 'y', 'z'])

        # test patch_val and slice_steps=2
        y = list(sorted(set(patch_val.y.astype(int))))
        x = list(sorted(set(patch_val.x.astype(int))))
        assert (int(y[1]) - int(y[0])) != SLICE_STEPS
        assert (int(x[1]) - int(x[0])) != SLICE_STEPS

        # test validation set is, at least, PER_VAL
        PER_VAL_CHK = len(set(patch_train.y))/(len(set(patch_train.y))+len(set(patch_val.y))) * 100
        assert round(PER_VAL_CHK,0) >= int(PER_VAL * 100)
        PER_VAL_CHK = len(set(patch_train.x))/(len(set(patch_train.x))+len(set(patch_val.x))) * 100
        assert round(PER_VAL_CHK,0) >= int(PER_VAL * 100)


def test_prepare_dutchf3_patch_train_and_test_sets():

    """check a complete run for the script in case further changes are needed
    """
    # setting a value to SLICE_STEPS as needed to test the values
    SLICE_STEPS = 1

    # use a temp dir that will be discarded at the end of the execution
    with tempfile.TemporaryDirectory() as tmpdirname:

        # saving the file to be used by the script
        label_file = tmpdirname + '/label_file.npy'
        np.save(label_file, ALINE)

        # stting the output directory to be used by the script
        output = tmpdirname + '/split'

        # calling the main function of the script without SLICE_STEPS, to check default value
        prep_dutchf3.split_patch_train_val(data_dir=tmpdirname, output_dir=output, label_file=label_file, 
        slice_steps=SLICE_STEPS, stride=STRIDE, patch_size=PATCH, per_val=PER_VAL,log_config=LOG_CONFIG)

        # reading the file and splitting the data
        patch_train = pd.read_csv(output + '/patch_train.txt', header=None, names=['row', 'a', 'b'])
        patch_train = pd.DataFrame(patch_train.row.str.split('_').tolist(), columns=['aline', 'x', 'y', 'z'])

        # reading the file and splitting the data
        patch_val = pd.read_csv(output + '/patch_val.txt', header=None, names=['row', 'a', 'b'])
        patch_val = pd.DataFrame(patch_val.row.str.split('_').tolist(), columns=['aline', 'x', 'y', 'z'])

        y_train = set(patch_train.y)
        x_train = set(patch_train.x)
        y_val = set(patch_val.y)
        x_val = set(patch_val.x)

        # The sets must not contain values common to both
        assert y_train & y_val == set()
        assert x_train & x_val == set()

        # test validation set is, at least, PER_VAL
        PER_VAL_CHK = len(set(patch_train.y))/(len(set(patch_train.y))+len(set(patch_val.y))) * 100
        assert round(PER_VAL_CHK,0) >= int(PER_VAL * 100)
        PER_VAL_CHK = len(set(patch_train.x))/(len(set(patch_train.x))+len(set(patch_val.x))) * 100
        assert round(PER_VAL_CHK,0) >= int(PER_VAL * 100)

def test_prepare_dutchf3_section_step_1():

    """check a complete run for the script in case further changes are needed
    """
    # setting a value to SLICE_STEPS as needed to test the values
    SLICE_STEPS = 1

    # use a temp dir that will be discarded at the end of the execution
    with tempfile.TemporaryDirectory() as tmpdirname:

        # saving the file to be used by the script
        label_file = tmpdirname + '/label_file.npy'
        np.save(label_file, ALINE)

        # stting the output directory to be used by the script
        output = tmpdirname + '/split'

        # calling the main function of the script without SLICE_STEPS, to check default value
        prep_dutchf3.split_section_train_val(data_dir=tmpdirname, output_dir=output, label_file=label_file,slice_steps=SLICE_STEPS, per_val=PER_VAL, log_config=LOG_CONFIG)

        # reading the file and splitting the data
        section_train = pd.read_csv(output + '/section_train.txt', header=None, names=['row'])
        section_train = pd.DataFrame(section_train.row.str.split('_').tolist(), columns=['aline', 'section'])

        section_val = pd.read_csv(output + '/section_val.txt', header=None, names=['row'])
        section_val = pd.DataFrame(section_val.row.str.split('_').tolist(), columns=['aline', 'section'])

        # test
        assert (float(section_train.section[1]) - float(section_train.section[0])) % float(SLICE_STEPS) == 0.0
        assert (float(section_val.section[1]) - float(section_val.section[0])) % float(SLICE_STEPS) == 0.0

        # test validation set is, at least, PER_VAL
        PER_VAL_CHK = len(section_val)/(len(section_val)+len(section_train)) * 100
        assert round(PER_VAL_CHK,0) >= int(PER_VAL * 100)

def test_prepare_dutchf3_section_step_2():

    """check a complete run for the script in case further changes are needed
    """
    # setting a value to SLICE_STEPS as needed to test the values
    SLICE_STEPS = 2

    # use a temp dir that will be discarded at the end of the execution
    with tempfile.TemporaryDirectory() as tmpdirname:

        # saving the file to be used by the script
        label_file = tmpdirname + '/label_file.npy'
        np.save(label_file, ALINE)

        # stting the output directory to be used by the script
        output = tmpdirname + '/split'

        # calling the main function of the script without SLICE_STEPS, to check default value
        prep_dutchf3.split_section_train_val(data_dir=tmpdirname, output_dir=output, label_file=label_file,
                                                slice_steps=SLICE_STEPS, per_val=PER_VAL, log_config=LOG_CONFIG)

        # reading the file and splitting the data
        section_train = pd.read_csv(output + '/section_train.txt', header=None, names=['row'])
        section_train = pd.DataFrame(section_train.row.str.split('_').tolist(), columns=['aline', 'section'])

        section_val = pd.read_csv(output + '/section_val.txt', header=None, names=['row'])
        section_val = pd.DataFrame(section_val.row.str.split('_').tolist(), columns=['aline', 'section'])

        # test
        assert (float(section_train.section[1]) - float(section_train.section[0])) % float(SLICE_STEPS) == 0.0
        assert (float(section_val.section[1]) - float(section_val.section[0])) % float(SLICE_STEPS) > 0.0

        # test validation set is, at least, PER_VAL
        PER_VAL_CHK = len(section_val)/(len(section_val)+len(section_train)) * 100
        assert round(PER_VAL_CHK,0) >= int(PER_VAL * 100)