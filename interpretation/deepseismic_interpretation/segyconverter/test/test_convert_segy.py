# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Test that the current scripts can run from the command line
"""
import os
import numpy as np
from deepseismic_interpretation.segyconverter import convert_segy
from deepseismic_interpretation.segyconverter.test import test_util
import pytest
import segyio

MAX_RANGE = 1
MIN_RANGE = 0
ERROR_EXIT_CODE = 99


@pytest.fixture(scope="class")
def segy_single_file(request):
    # setup code
    # create segy file
    inlinefile = "./inlinesortsample.segy"
    test_util.create_segy_file(
        lambda il, xl: not ((il < 20 and xl < 125) or (il > 40 and xl > 250)),
        inlinefile,
        segyio.TraceSortingFormat.INLINE_SORTING,
    )

    # inject class variables
    request.cls.testfile = inlinefile
    yield

    # teardown code
    os.remove(inlinefile)


@pytest.mark.usefixtures("segy_single_file")
class TestConvertSEGY:

    testfile = None  # Set by segy_file fixture

    def test_convert_segy_generates_single_npy(self, tmpdir):
        # Setup
        prefix = "volume1"
        input_file = self.testfile
        output_dir = tmpdir.strpath
        metadata_only = False
        iline = 189
        xline = 193
        cube_size = -1
        stride = 128
        normalize = True
        clip = True
        inputpath = ""

        # Test
        convert_segy.main(
            input_file, output_dir, prefix, iline, xline, metadata_only, stride, cube_size, normalize, clip
        )

        # Validate
        npy_files = test_util.get_npy_files(tmpdir.strpath)
        assert len(npy_files) == 1

        min_val, max_val = _get_min_max(tmpdir.strpath)
        assert min_val >= MIN_RANGE
        assert max_val <= MAX_RANGE

    def test_convert_segy_generates_multiple_npy_files(self, tmpdir):
        """
        Run process_all_files and checks that it returns with 0 exit code
        :param function filedir: fixture for setup and cleanup
        """

        # Setup
        prefix = "volume1"
        input_file = self.testfile
        output_dir = tmpdir.strpath
        metadata_only = False
        iline = 189
        xline = 193
        cube_size = 128
        stride = 128
        normalize = True
        inputpath = ""
        clip = True
        # Test
        convert_segy.main(
            input_file, output_dir, prefix, iline, xline, metadata_only, stride, cube_size, normalize, clip
        )

        # Validate
        npy_files = test_util.get_npy_files(tmpdir.strpath)
        assert len(npy_files) == 2

    def test_convert_segy_normalizes_data(self, tmpdir):
        """
        Run process_all_files and checks that it returns with 0 exit code
        :param function filedir: fixture for setup and cleanup
        """

        # Setup
        prefix = "volume1"
        input_file = self.testfile
        output_dir = tmpdir.strpath
        metadata_only = False
        iline = 189
        xline = 193
        cube_size = 128
        stride = 128
        normalize = True
        inputpath = ""
        clip = True

        # Test
        convert_segy.main(
            input_file, output_dir, prefix, iline, xline, metadata_only, stride, cube_size, normalize, clip
        )

        # Validate
        npy_files = test_util.get_npy_files(tmpdir.strpath)
        assert len(npy_files) == 2
        min_val, max_val = _get_min_max(tmpdir.strpath)
        assert min_val >= MIN_RANGE
        assert max_val <= MAX_RANGE

    def test_convert_segy_clips_data(self, tmpdir):
        """
        Run process_all_files and checks that it returns with 0 exit code
        :param function filedir: fixture for setup and cleanup
        """

        # Setup
        prefix = "volume1"
        input_file = self.testfile
        output_dir = tmpdir.strpath
        metadata_only = False
        iline = 189
        xline = 193
        cube_size = 128
        stride = 128
        normalize = False
        inputpath = ""
        clip = True

        # Test
        convert_segy.main(
            input_file, output_dir, prefix, iline, xline, metadata_only, stride, cube_size, normalize, clip
        )

        # Validate
        expected_max = 35.59
        expected_min = -35.59
        npy_files = test_util.get_npy_files(tmpdir.strpath)
        assert len(npy_files) == 2
        min_val, max_val = _get_min_max(tmpdir.strpath)
        assert expected_min == pytest.approx(min_val, rel=1e-3)
        assert expected_max == pytest.approx(max_val, rel=1e-3)

    def test_convert_segy_copies_exact_data_with_no_normalization(self, tmpdir):
        """
        Run process_all_files and checks that it returns with 0 exit code
        :param function filedir: fixture for setup and cleanup
        """

        # Setup
        prefix = "volume1"
        input_file = self.testfile
        output_dir = tmpdir.strpath
        metadata_only = False
        iline = 189
        xline = 193
        cube_size = 128
        stride = 128
        normalize = False
        inputpath = ""
        clip = False

        # Test
        convert_segy.main(
            input_file, output_dir, prefix, iline, xline, metadata_only, stride, cube_size, normalize, clip
        )

        # Validate
        expected_max = 1039.8
        expected_min = -1039.8
        npy_files = test_util.get_npy_files(tmpdir.strpath)
        assert len(npy_files) == 2
        min_val, max_val = _get_min_max(tmpdir.strpath)
        assert expected_min == pytest.approx(min_val, rel=1e-3)
        assert expected_max == pytest.approx(max_val, rel=1e-3)


def _get_min_max(outputdir):
    """
    Check # of npy files in directory
    :param str outputdir: directory to check for npy files
    :returns: min_val, max_val of values in npy files
    :rtype: int, int
    """
    min_val = 0
    max_val = 0
    npy_files = test_util.get_npy_files(outputdir)
    for file in npy_files:
        data = np.load(os.path.join(outputdir, file))
        this_min = np.amin(data)
        this_max = np.amax(data)
        if this_min < min_val:
            min_val = this_min
        if this_max > max_val:
            max_val = this_max
    return min_val, max_val
