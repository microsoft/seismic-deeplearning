# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Test the extract functions against a variety of SEGY files and trace_header scenarioes
"""
import os
import pytest
import numpy as np
import pandas as pd
from deepseismic_interpretation.segyconverter.utils import segyextract
from deepseismic_interpretation.segyconverter.test import test_util
import segyio
import json

FILENAME = "./normalsegy.segy"
PREFIX = "normal"


@pytest.fixture(scope="class")
def segy_all_files(request):
    # setup code
    # create segy file
    normal_filename = "./normalsegy.segy"
    test_util.create_segy_file(lambda il, xl: True, normal_filename)
    request.cls.control_file = normal_filename  # Set by segy_file fixture

    inline_filename = "./inlineerror.segy"
    test_util.create_segy_file(
        lambda il, xl: not ((il < 20 and xl < 125) or (il > 40 and xl > 250)),
        inline_filename,
        segyio.TraceSortingFormat.INLINE_SORTING,
    )
    request.cls.inline_sort_file = inline_filename  # Set by segy_file fixture

    xline_filename = "./xlineerror.segy"
    test_util.create_segy_file(
        lambda il, xl: not ((il < 20 and xl < 125) or (il > 40 and xl > 250)),
        xline_filename,
        segyio.TraceSortingFormat.CROSSLINE_SORTING,
    )
    request.cls.crossline_sort_file = xline_filename  # Set by segy_file fixture

    hole_filename = "./hole.segy"
    test_util.create_segy_file(
        lambda il, xl: not ((20 < il < 30) and (150 < xl < 250)),
        hole_filename,
        segyio.TraceSortingFormat.INLINE_SORTING,
    )
    request.cls.hole_file = hole_filename  # Set by segy_file fixture

    yield

    # teardown code
    os.remove(normal_filename)
    os.remove(inline_filename)
    os.remove(xline_filename)
    os.remove(hole_filename)


@pytest.mark.usefixtures("segy_all_files")
class TestSEGYExtract:

    control_file = None  # Set by segy_file fixture
    inline_sort_file = None  # Set by segy_file fixture
    crossline_sort_file = None  # Set by segy_file fixture
    hole_file = None  # Set by segy_file fixture

    @pytest.mark.parametrize(
        "filename, trace_count, first_inline, inline_count, first_xline, xline_count, depth",
        [
            ("./normalsegy.segy", 8000, 10, 40, 100, 200, 10),
            ("./inlineerror.segy", 7309, 10, 40, 125, 200, 10),
            ("./xlineerror.segy", 7309, 10, 40, 125, 200, 10),
            ("./hole.segy", 7109, 10, 40, 100, 200, 10),
        ],
    )
    def test_get_segy_metadata_should_return_correct_metadata(
        self, filename, trace_count, first_inline, inline_count, first_xline, xline_count, depth
    ):
        """
        Check that get_segy_metadata can correctly identify the sorting from the trace headers
        :param dict tmpdir: pytest fixture for local test directory cleanup
        :param str filename: SEG-Y filename
        :param int inline: byte location for inline
        :param int xline: byte location for crossline
        :param int depth: number of samples
        """
        # setup
        inline_byte_loc = 189
        xline_byte_loc = 193

        # test
        fast_indexes, slow_indexes, trace_headers, sample_size = segyextract.get_segy_metadata(
            filename, inline_byte_loc, xline_byte_loc
        )

        # validate
        assert sample_size == depth
        assert len(trace_headers) == trace_count
        assert len(fast_indexes) == inline_count
        assert len(slow_indexes) == xline_count

        # Check fast direction
        assert trace_headers["slow"][0] == first_xline
        assert trace_headers["fast"][0] == first_inline

    @pytest.mark.parametrize(
        "filename,inline,xline,depth",
        [
            ("./normalsegy.segy", 40, 200, 10),
            ("./inlineerror.segy", 40, 200, 10),
            ("./xlineerror.segy", 40, 200, 10),
            ("./hole.segy", 40, 200, 10),
        ],
    )
    def test_process_segy_data_should_create_cube_size_equal_to_segy(self, tmpdir, filename, inline, xline, depth):
        """
        Create single npy file for segy and validate size
        :param dict tmpdir: pytest fixture for local test directory cleanup
        :param str filename: SEG-Y filename
        :param int inline: byte location for inline
        :param int xline: byte location for crossline
        :param int depth: number of samples
        """
        segyextract.process_segy_data_into_single_array(filename, tmpdir.strpath, PREFIX)

        npy_files = test_util.get_npy_files(tmpdir.strpath)
        assert len(npy_files) == 1

        data = np.load(os.path.join(tmpdir.strpath, npy_files[0]))
        assert len(data.shape) == 3
        assert data.shape[0] == inline
        assert data.shape[1] == xline
        assert data.shape[2] == depth

    def test_process_segy_data_should_write_npy_files_for_n_equals_128_stride_64(self, tmpdir):
        """
        Break data up into size n=128 size blocks and validate against original segy
        file. This size of block causes the code to write 1 x 4 npy files
        :param function tmpdir: pytest fixture for local test directory cleanup
        """
        # setup
        n_points = 128
        stride = 64

        # test
        segyextract.process_segy_data(FILENAME, tmpdir.strpath, PREFIX, n_points=n_points, stride=stride)

        # validate
        _output_npy_files_are_correct_for_cube_size(4, 128, tmpdir.strpath)

    def test_process_segy_data_should_write_npy_files_for_n_equals_128(self, tmpdir):
        """
        Break data up into size n=128 size blocks and validate against original segy
        file. This size of block causes the code to write 1 x 4 npy files
        :param function tmpdir: pytest fixture for local test directory cleanup
        """
        # setup
        n_points = 128

        # test
        segyextract.process_segy_data(FILENAME, tmpdir.strpath, PREFIX)

        # validate
        npy_files = _output_npy_files_are_correct_for_cube_size(2, 128, tmpdir.strpath)

        full_volume_from_file = test_util.build_volume(n_points, npy_files, tmpdir.strpath)

        # Validate contents of volume
        _compare_variance(FILENAME, PREFIX, full_volume_from_file, tmpdir.strpath)

    def test_process_segy_data_should_write_npy_files_for_n_equals_64(self, tmpdir):
        """
        Break data up into size n=64 size blocks and validate against original segy
        file. This size of block causes the code to write 1 x 8 npy files
        :param function tmpdir: pytest fixture for local test directory cleanup
        """
        # setup

        n_points = 64
        expected_file_count = 4
        # test
        segyextract.process_segy_data(FILENAME, tmpdir.strpath, PREFIX, n_points=n_points, stride=n_points)

        # validate
        npy_files = _output_npy_files_are_correct_for_cube_size(expected_file_count, n_points, tmpdir.strpath)

        full_volume_from_file = test_util.build_volume(n_points, npy_files, tmpdir.strpath)

        # Validate contents of volume
        _compare_variance(FILENAME, PREFIX, full_volume_from_file, tmpdir.strpath)

    def test_process_segy_data_should_write_npy_files_for_n_equals_16(self, tmpdir):
        """
        Break data up into size n=16 size blocks and validate against original segy
        file. This size of block causes the code to write 2 x 4 x 32 npy files.
        :param function tmpdir: pytest fixture for local test directory cleanup
        """
        # setup
        n_points = 16

        # test
        segyextract.process_segy_data(FILENAME, tmpdir.strpath, PREFIX, n_points=n_points, stride=n_points)

        # validate
        npy_files = _output_npy_files_are_correct_for_cube_size(39, 16, tmpdir.strpath)

        full_volume_from_file = test_util.build_volume(n_points, npy_files, tmpdir.strpath)
        _compare_variance(FILENAME, PREFIX, full_volume_from_file, tmpdir.strpath)

    def test_process_npy_file_should_have_same_content_as_segy(self, tmpdir):
        """
        Check the actual content of a npy file generated from the segy
        :param function tmpdir: pytest fixture for local test directory cleanup
        """
        segyextract.process_segy_data_into_single_array(FILENAME, tmpdir.strpath, PREFIX)

        npy_files = test_util.get_npy_files(tmpdir.strpath)
        assert len(npy_files) == 1

        data = np.load(os.path.join(tmpdir.strpath, npy_files[0]))
        _compare_output_to_segy(FILENAME, data, 40, 200, 10)

    def test_remove_duplicates_should_keep_order(self):
        # setup
        list_with_dups = [1, 2, 3, 3, 5, 8, 4, 2]
        # test
        result = segyextract._remove_duplicates(list_with_dups)
        # validate
        expected_result = [1, 2, 3, 5, 8, 4]
        assert all([a == b for a, b in zip(result, expected_result)])

    def test_identify_fast_direction_should_handle_xline_sequence_1(self):
        # setup
        df = pd.DataFrame({"i": [101, 102, 102, 102, 103, 103], "j": [301, 301, 302, 303, 301, 302]})
        # test
        segyextract._identify_fast_direction(df, "fast", "slow")
        # validate
        assert df.keys()[0] == "fast"
        assert df.keys()[1] == "slow"

    def test_identify_fast_direction_should_handle_xline_sequence_2(self):
        # setup
        df = pd.DataFrame({"i": [101, 102, 102, 102, 102, 102], "j": [301, 301, 302, 303, 304, 305]})
        # test
        segyextract._identify_fast_direction(df, "fast", "slow")
        # validate
        assert df.keys()[0] == "fast"
        assert df.keys()[1] == "slow"


def _output_npy_files_are_correct_for_cube_size(expected_count, cube_size, outputdir):
    """
    Check # of npy files in directory
    :param int expected_count: expected # of npy files
    :param str outputdir: directory to check for npy files
    :param int cube_size: size of cube array
    :returns: npy_files in outputdir
    :rtype: list
    """
    npy_files = test_util.get_npy_files(outputdir)
    assert len(npy_files) == expected_count

    data = np.load(os.path.join(outputdir, npy_files[0]))
    assert len(data.shape) == 3
    assert data.shape.count(cube_size) == 3

    return npy_files


def _compare_output_to_segy(filename, data, fast_size, slow_size, depth):
    """
    Compares each trace in the segy file to the data volume that
    was generated from the npy file. This only works when a single npy
    is created from a cuboid SEGY. If the dimensions are not aligned

    :param str filename: path to segy file
    :param nparray data: data read in from npy files
    """
    with segyio.open(filename, ignore_geometry=True) as segy_file:
        segy_file.mmap()
        segy_sum = np.float32(0.0)
        npy_sum = np.float32(0.0)
        # Validate that each trace in the segy file is represented in the npy files
        # Sum traces in segy and npy to ensure they are correct
        for j in range(0, fast_size):  # Fast
            for i in range(0, slow_size):  # Slow
                trace = segy_file.trace[i + (j * slow_size)]
                data_trace = data[j, i, :]
                assert all([a == b for a, b in zip(trace, data_trace)]), f"Unmatched trace at {j}:{i}"
                segy_sum += np.sum(trace, dtype=np.float32)
                npy_sum += np.sum(data_trace, dtype=np.float32)
        assert segy_sum == npy_sum


def _compare_variance(filename, prefix, data, outputdir):
    """
    Compares the standard deviation calculated from the full volume to the
    standard deviation calculated while creating the npy files

    :param str filename: path to segy file
    :param str prefix: prefix used to find files
    :param nparray data: data read in from npy files
    :param str outputdir: location of npy files
    """
    with segyio.open(filename, ignore_geometry=True) as segy_file:
        segy_file.mmap()
        segy_stddev = np.sqrt(np.var(data))

    # Check statistics file generated from segy
    with open(os.path.join(outputdir, prefix + "_stats.json"), "r") as f:
        metadatastr = f.read()

    metadata = json.loads(metadatastr)
    stddev = float(metadata["stddev"])

    assert round(stddev) == round(segy_stddev)
