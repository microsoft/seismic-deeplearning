# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Test data normalization
"""
import os
import numpy as np
import utils.dataprep as dataprep
import pytest
import test_util
import segyio

INPUT_FOLDER = './contrib/segyconverter/test/test_data'
MAX_RANGE = 1
MIN_RANGE = 0
K = 12


class TestNormalizeCube:

    testcube = None  # Set by npy_files fixture

    def test_normalize_cube_returns_normalized_values(self):
        """
            Test method that normalize one cube by checking if normalized 
            values are within [min, max] range.
        """
        trace = np.linspace(-1, 1, 100, True, dtype=np.single)
        cube = np.ones((100, 50, 100)) * trace * 500
        # Add values to clip
        cube[40,25,50] = 700
        cube[70,30,70] = -700
        mean = np.mean(cube)
        variance = np.var(cube)
        stddev = np.sqrt(variance)
        min_clip, max_clip, scale = dataprep.compute_statistics(stddev, mean, MAX_RANGE, K)
        norm_block = dataprep.normalize_cube(cube, min_clip, max_clip, scale, MIN_RANGE, MAX_RANGE)
        assert np.amax(norm_block) <= MAX_RANGE
        assert np.amin(norm_block) >= MIN_RANGE

    def test_clip_cube_returns_clipped_values(self):
        """
            Test method that clip one cube by checking if clipped 
            values are within [min_clip, max_clip] range.
        """
        trace = np.linspace(-1, 1, 100, True, dtype=np.single)
        cube = np.ones((100, 50, 100)) * trace * 500
        # Add values to clip
        cube[40, 25, 50] = 700
        cube[70, 30, 70] = -700
        mean = np.mean(cube)
        variance = np.var(cube)
        stddev = np.sqrt(variance)
        min_clip, max_clip, scale = dataprep.compute_statistics(stddev, mean, MAX_RANGE, K)
        clipped_block = dataprep.clip_cube(cube, min_clip, max_clip)
        assert np.amax(clipped_block) <= max_clip
        assert np.amin(clipped_block) >= min_clip

    def test_norm_value_is_correct(self):
        # Check if normalized value is calculated correctly
        min_clip = -18469.875210304104
        max_clip = 18469.875210304104
        scale = 2.707110872741882e-05
        input_value = 2019
        expected_norm_value = 0.5546565685206586
        norm_v = dataprep.norm_value(input_value, min_clip, max_clip, MIN_RANGE, MAX_RANGE, scale)
        assert norm_v == pytest.approx(expected_norm_value, rel=1e-3)

    def test_clip_value_is_correct(self):
        # Check if normalized value is calculated correctly
        min_clip = -18469.875210304104
        max_clip = 18469.875210304104
        input_value = 2019
        expected_clipped_value = 2019
        clipped_v = dataprep.clip_value(input_value, min_clip, max_clip)
        assert clipped_v == pytest.approx(expected_clipped_value, rel=1e-3)

    def test_norm_value_on_cube_is_within_range(self):
        # Check if normalized value is within [MIN_RANGE, MAX_RANGE]
        trace = np.linspace(-1, 1, 100, True, dtype=np.single)
        cube = np.ones((100, 50, 100)) * trace * 500
        cube[40, 25, 50] = 7000
        cube[70, 30, 70] = -7000
        variance = np.var(cube)
        stddev = np.sqrt(variance)
        mean = np.mean(cube)
        v = cube[10, 40, 5]
        min_clip, max_clip, scale = dataprep.compute_statistics(stddev, mean, MAX_RANGE, K)
        norm_v = dataprep.norm_value(v, min_clip, max_clip, MIN_RANGE, MAX_RANGE, scale)
        assert norm_v <= MAX_RANGE
        assert norm_v >= MIN_RANGE

        pytest.raises(Exception, dataprep.norm_value, v, min_clip * 10, max_clip * 10, 
                      MIN_RANGE, MAX_RANGE, scale * 10)

    def test_clipped_value_on_cube_is_within_range(self):
        # Check if clipped value is within [min_clip, max_clip]
        trace = np.linspace(-1, 1, 100, True, dtype=np.single)
        cube = np.ones((100, 50, 100)) * trace * 500
        cube[40, 25, 50] = 7000
        cube[70, 30, 70] = -7000
        variance = np.var(cube)
        mean = np.mean(cube)
        stddev = np.sqrt(variance)
        v = cube[10, 40, 5]
        min_clip, max_clip, scale = dataprep.compute_statistics(stddev, mean, MAX_RANGE, K)
        clipped_v = dataprep.clip_value(v, min_clip, max_clip)
        assert clipped_v <= max_clip
        assert clipped_v >= min_clip

    def test_compute_statistics(self):
        # Check if statistics are calculated correctly for provided stddev, max_range and k values
        expected_min_clip = -138.693888
        expected_max_clip = 138.693888
        expected_scale = 0.003605061529459755
        mean = 0
        stddev = 11.557824
        min_clip, max_clip, scale = dataprep.compute_statistics(stddev, mean, MAX_RANGE, K)
        assert expected_min_clip == pytest.approx(min_clip, rel=1e-3)
        assert expected_max_clip == pytest.approx(max_clip, rel=1e-3)
        assert expected_scale == pytest.approx(scale, rel=1e-3)
        # Testing division by zero
        pytest.raises(Exception, dataprep.compute_statistics, stddev, MAX_RANGE, 0)
        pytest.raises(Exception, dataprep.compute_statistics, 0, MAX_RANGE, 0)

    def test_apply_should_clip_and_normalize_data(self):
        # Check that apply method will clip and normalize the data
        trace = np.linspace(-1, 1, 100, True, dtype=np.single)
        cube = np.ones((100, 50, 100)) * trace * 500
        cube[40, 25, 50] = 7000
        cube[70, 30, 70] = -7000
        variance = np.var(cube)
        stddev = np.sqrt(variance)
        mean = np.mean(cube)

        norm_block = dataprep.apply(cube, stddev, mean, K, MIN_RANGE, MAX_RANGE)
        assert np.amax(norm_block) <= MAX_RANGE
        assert np.amin(norm_block) >= MIN_RANGE

        norm_block = dataprep.apply(cube, stddev, mean, K, MIN_RANGE, MAX_RANGE, clip=False)
        assert np.amax(norm_block) <= MAX_RANGE
        assert np.amin(norm_block) >= MIN_RANGE

        pytest.raises(Exception, dataprep.apply, cube, stddev, 0, MIN_RANGE, MAX_RANGE)
        pytest.raises(Exception, dataprep.apply, cube, 0, K, MIN_RANGE, MAX_RANGE)

        invalid_cube = np.empty_like(cube)
        invalid_cube[:] = np.nan
        pytest.raises(Exception, dataprep.apply, invalid_cube, stddev, 0, MIN_RANGE, MAX_RANGE)

    def test_apply_should_clip_data(self):
        # Check that apply method will clip the data
        trace = np.linspace(-1, 1, 100, True, dtype=np.single)
        cube = np.ones((100, 50, 100)) * trace * 500
        cube[40, 25, 50] = 7000
        cube[70, 30, 70] = -7000
        variance = np.var(cube)
        stddev = np.sqrt(variance)
        mean = np.mean(cube)
        min_clip, max_clip, _ = dataprep.compute_statistics(stddev, mean, MAX_RANGE, K)
        norm_block = dataprep.apply(cube, stddev, mean, K, MIN_RANGE, MAX_RANGE, clip=True,
                                    normalize=False)
        assert np.amax(norm_block) <= max_clip
        assert np.amin(norm_block) >= min_clip

        invalid_cube = np.empty_like(cube)
        invalid_cube[:] = np.nan
        pytest.raises(Exception, dataprep.apply, invalid_cube, stddev, 0, MIN_RANGE, MAX_RANGE,
                      clip=True, normalize=False)
