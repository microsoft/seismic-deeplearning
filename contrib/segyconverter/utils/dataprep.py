# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Utility Script to normalize one cube
"""

import numpy as np


def compute_statistics(stddev: float, mean: float, max_range: float, k: int):
    """
        Compute min_clip, max_clip and scale values based on provided stddev, max_range and k values
        :param stddev: standard deviation value
        :param max_range: maximum value range
        :param k: number of standard deviation to be used in normalization
        :returns: min_clip, max_clip, scale: computed values
        :rtype: float, float, float
    """
    min_clip = mean - k * stddev
    max_clip = mean + k * stddev
    scale = max_range / (max_clip - min_clip)
    return min_clip, max_clip, scale


def clip_value(v: float, min_clip: float, max_clip: float):
    """
        Clip seismic voxel value
        :param min_clip: minimum value used for clipping
        :param max_clip: maximum value used for clipping
        :returns: clipped value, must be within [min_clip, max_clip]
        :rtype: float
    """
    # Clip value
    if v > max_clip:
        v = max_clip
    if v < min_clip:
        v = min_clip
    return v


def norm_value(v: float, min_clip: float, max_clip: float, min_range: float, max_range: float, scale: float):
    """
        Normalize seismic voxel value to be within [min_range, max_clip] according to
        statisctics computed previously
        :param v: value to be normalized
        :param min_clip: minimum value used for clipping
        :param max_clip: maximum value used for clipping
        :param min_range: minium range value
        :param max_range: maximum range value
        :param scale: scale value to be used for normalization
        :returns: normalized value, must be within [min_range, max_range]
        :rtype: float
    """
    offset = -1 * min_clip  # Normalizing - set values between 0 and 1
    # Clip value
    v = clip_value(v, min_clip, max_clip)
    # Scale value
    v = (v + offset) * scale
    # This value should ALWAYS be between min_range and max_range here
    if v > max_range or v < min_range:
        raise Exception('normalized value should be within [{0},{1}].\
             The value was: {2}'.format(min_range, max_range, v))
    return v


def normalize_cube(cube: np.array, min_clip: float, max_clip: float, scale: float,
                   min_range: float, max_range: float):
    """
        Normalize cube according to statistics. Normalization implies in clipping and normalize cube.
        :param cube: 3D array to be normalized
        :param min_clip: minimum value used for clipping
        :param max_clip: maximum value used for clipping
        :param min_range: minium range value
        :param max_range: maximum range value
        :param scale: scale value to be used for normalization
        :returns: normalized 3D array
        :rtype: numpy array
    """
    # Define function for normalization
    vfunc = np.vectorize(norm_value)
    # Normalize cube
    norm_cube = vfunc(cube, min_clip=min_clip, max_clip=max_clip, min_range=min_range, max_range=max_range,
                      scale=scale)
    return norm_cube


def clip_cube(cube: np.array, min_clip: float, max_clip: float):
    """
        Clip cube values according to statistics
        :param min_clip: minimum value used for clipping
        :param max_clip: maximum value used for clipping
        :returns: clipped 3D array
        :rtype: numpy array
    """
    # Define function for normalization
    vfunc = np.vectorize(clip_value)
    clip_cube = vfunc(cube, min_clip=min_clip, max_clip=max_clip)
    return clip_cube


def apply(cube: np.array, stddev: float, mean: float, k: float, min_range: float, max_range: float,
          clip=True, normalize=True):
    """
    Preapre data according to provided parameters. This method will compute satistics and can
    normalize&clip, just clip, or leave the data as is. 
    :param cube: 3D array to be normalized
    :param stddev: standard deviation value
    :param k: number of standard deviation to be used in normalization
    :param min_range: minium range value
    :param max_range: maximum range value
    :param clip: flag to turn on/off clip
    :param normalize: flag to turn on/off normalization.
    :returns: processed 3D array
    :rtype: numpy array
    """
    if np.isnan(np.min(cube)):
        raise Exception("Cube has NaN value")
    if stddev == 0.0:
        raise Exception("Standard deviation must not be zero")
    if k == 0:
        raise Exception("k must not be zero")

    # Compute statistics
    min_clip, max_clip, scale = compute_statistics(stddev=stddev, mean=mean, k=k, max_range=max_range)

    if (clip and normalize) or normalize:
        # Normalize&clip cube. Note that it is not possible to normalize data without 
        # applying clip operation
        print("Normalizing and Clipping File")
        return normalize_cube(cube=cube, min_clip=min_clip, max_clip=max_clip, scale=scale, 
                              min_range=min_range, max_range=max_range)
    elif clip:
        # Only clip values
        print("Clipping File")
        return clip_cube(cube=cube, min_clip=min_clip, max_clip=max_clip)
