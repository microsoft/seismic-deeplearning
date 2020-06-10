# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Methods for processing segy files that do not include well formed geometry. In these cases, segyio
cannot infer the 3D volume of data from the traces so this module needs to do that manually
"""

import os
import math
import segyio
import pandas as pd
import numpy as np
import json

FAST = "fast"
SLOW = "slow"
DEFAULT_VALUE = 255


def get_segy_metadata(input_file, iline, xline):
    """
    Loads segy file and uses the input inline and crossline byte values to load
    the trace headers. It determines which inline or crossline the traces
    start with. SEGY files can be non standard and use other byte location for
    these values. In that case, the data from this method will be erroneous. It
    is up to the user to figure out which numbers to use by reading the SEGY text
    header and finding the byte offsets visually.

    :param str input_file: path to segy file
    :param int iline: inline byte position
    :param int xline: crossline byte position
    :returns: fast_distinct, slow_distinct, trace_headers, samplesize
    :rtype: list, DataFrame, DataFrame, int
    """

    with segyio.open(input_file, ignore_geometry=True) as segy_file:
        segy_file.mmap()
        # Initialize df with trace id as index and headers as columns
        trace_headers = pd.DataFrame(index=range(0, segy_file.tracecount), columns=["i", "j"])

        # Fill dataframe with all trace headers values
        trace_headers["i"] = segy_file.attributes(iline)
        trace_headers["j"] = segy_file.attributes(xline)

        _identify_fast_direction(trace_headers, FAST, SLOW)

        samplesize = len(segy_file.samples)
        fast_distinct = _remove_duplicates(trace_headers[FAST])
        slow_distinct = np.unique(trace_headers[SLOW])
    return fast_distinct, slow_distinct, trace_headers, samplesize


def process_segy_data_into_single_array(input_file, output_dir, prefix, iline=189, xline=193):
    """
    Open segy file and write all data to single npy array
    :param str input_file: path to segyfile
    :param str output_dir: path to directory where npy files will be written/rewritten
    :param str prefix: prefix to use when writing npy files
    :param int iline: iline header byte location
    :param int xline: crossline header byte location
    :returns: 3 dimentional numpy array of SEGY data
    :rtype: nparray
    """
    fast_distinct, slow_distinct, trace_headers, sampledepth = get_segy_metadata(input_file, iline, xline)
    with segyio.open(input_file, ignore_geometry=True) as segy_file:
        segy_file.mmap()

        fast_line_space = abs(fast_distinct[1] - fast_distinct[0])

        slow_line_space = abs(slow_distinct[0] - slow_distinct[1])
        sample_size = len(segy_file.samples)
        layer_fastmax = max(fast_distinct)
        layer_fastmin = min(fast_distinct)
        layer_slowmax = max(slow_distinct)
        layer_slowmin = min(slow_distinct)
        layer_trace_ids = trace_headers[
            (trace_headers.fast >= layer_fastmin)
            & (trace_headers.fast <= layer_fastmax)
            & (trace_headers.slow >= layer_slowmin)
            & (trace_headers.slow <= layer_slowmax)
        ]

        block = np.full((len(fast_distinct), len(slow_distinct), sampledepth), DEFAULT_VALUE, dtype=np.float32)
        for _, row in layer_trace_ids.iterrows():
            block[
                (row[FAST] - layer_fastmin) // fast_line_space,
                (row[SLOW] - layer_slowmin) // slow_line_space,
                0:sample_size,
            ] = segy_file.trace[row.name]

        np.save(
            os.path.join(output_dir, "{}_{}_{}_{:05d}".format(prefix, fast_distinct[0], slow_distinct[0], 0)), block
        )
        variance = np.var(block)
        stddev = np.sqrt(variance)
        mean = np.mean(block)

        with open(os.path.join(output_dir, prefix + "_stats.json"), "w") as f:
            f.write(json.dumps({"stddev": str(stddev), "mean": str(mean)}))
        print("Npy files written: 1")
    return block


def process_segy_data(input_file, output_dir, prefix, iline=189, xline=193, n_points=128, stride=128):
    """
    Open segy file and write all numpy array files to disk
    :param str input_file: path to segyfile
    :param str output_dir: path to directory where npy files will be written/rewritten
    :param str prefix: prefix to use when writing npy files
    :param int iline: iline header byte location
    :param int xline: crossline header byte location
    :param int n_points: output cube size
    :param int stride: stride when writing data
    """
    fast_indexes, slow_indexes, trace_headers, _ = get_segy_metadata(input_file, iline, xline)
    with segyio.open(input_file, ignore_geometry=True) as segy_file:
        segy_file.mmap()
        # Global variance of segy data
        variance = 0
        mean = 0
        sample_count = 0
        filecount = 0
        block_size = n_points ** 3
        for block, i, j, k in _generate_all_blocks(
            segy_file, n_points, stride, fast_indexes, slow_indexes, trace_headers
        ):
            # Getting global variance as sum of local variance
            if variance == 0:
                # init
                variance = np.var(block)
                mean = np.mean(block)
                sample_count = block_size
            else:
                new_avg = np.mean(block)
                new_variance = np.var(block)
                variance = _parallel_variance(mean, sample_count, variance, new_avg, block_size, new_variance)
                mean = ((mean * sample_count) + np.sum(block)) / (sample_count + block_size)
                sample_count += block_size

            np.save(os.path.join(output_dir, "{}_{}_{}_{:05d}".format(prefix, i, j, k)), block)
            filecount += 1

        stddev = np.sqrt(variance)
        with open(os.path.join(output_dir, prefix + "_stats.json"), "w") as f:
            f.write(json.dumps({"stddev": stddev, "mean": mean}))
        print("Npy files written: {}".format(filecount))


def process_segy_data_column(input_file, output_dir, prefix, i, j, iline=189, xline=193, n_points=128, stride=128):
    """
    Open segy file and write one column of npy files to disk
    :param str input_file: segy file path
    :param str output_dir: local output directory for npy files
    :param str prefix: naming prefix for npy files
    :param int i: index for column data to extract
    :param int j: index for column data to extractc
    :param int iline: header byte location for inline
    :param int xline: header byte location for crossline
    :param int n_points: size of cube
    :param int stride: stride for generating cubes
    """
    fast_indexes, slow_indexes, trace_headers, _ = get_segy_metadata(input_file, iline, xline)

    with segyio.open(input_file, ignore_geometry=True) as segy_file:
        segy_file.mmap()
        filecount = 0
        for block, i, j, k in _generate_column_blocks(
            segy_file, n_points, stride, i, j, fast_indexes, slow_indexes, trace_headers
        ):
            np.save(os.path.join(output_dir, "{}_{}_{}_{}".format(prefix, i, j, k)), block)
            filecount += 1
        print("Files written: {}".format(filecount))


def _parallel_variance(avg_a, count_a, var_a, avg_b, count_b, var_b):
    """
    Calculate the new variance based on previous calcuated variance
    :param float avg_a: overall average
    :param float count_a: overall count
    :param float var_a: current variance
    :param float avg_b: ne average
    :param float count_b: current count
    :param float var_b: current variance
    :returns: new variance
    :rtype: float
    """
    delta = avg_b - avg_a
    m_a = var_a * (count_a - 1)
    m_b = var_b * (count_b - 1)
    M2 = m_a + m_b + delta ** 2 * count_a * count_b / (count_a + count_b)
    return M2 / (count_a + count_b - 1)


def _identify_fast_direction(trace_headers, fastlabel, slowlabel):
    """
    Returns the modified dataframe with columns labelled as 'fast' and 'slow'
    Uses the count of changes in indexes for both columns to determine which one is the fast index

    :param DataFrame trace_headers: dataframe with two columns
    :param str fastlabel: key label for the fast index
    :param str slowlabel: key label for the slow index
    """
    j_count = 0
    i_count = 0
    last_trace = 0
    slope_run = 5
    for trace in trace_headers["j"][0:slope_run]:
        if not last_trace == trace:
            j_count += 1
            last_trace = trace

    last_trace = 0
    for trace in trace_headers["i"][0:slope_run]:
        if not last_trace == trace:
            i_count += 1
            last_trace = trace
    if i_count < j_count:
        trace_headers.columns = [fastlabel, slowlabel]
    else:
        trace_headers.columns = [slowlabel, fastlabel]


def _remove_duplicates(list_of_elements):
    """
    Remove duplicates from a list but maintain the order
    :param list list_of_elements: list to be deduped
    :returns: list containing a distinct list of elements
    :rtype: list
    """
    seen = set()
    return [x for x in list_of_elements if not (x in seen or seen.add(x))]


def _get_trace_column(n_lines, i, j, trace_headers, fast_distinct, slow_distinct, segyfile):
    """
    :param int n_lines: number of voxels to extract in each dimension
    :param int i: fast index anchor for origin of column
    :param int j: slow index anchor for origin of column
    :param DataFrame trace_headers: DataFrame of all trace headers
    :param list fast_distinct: list of distinct fast headers
    :param list slow_distinct: list of distinct slow headers
    :param segyio.file segyfile: segy file object previously opened using segyio
    :returns: thiscolumn, layer_fastmin, layer_slowmin
    :rtype: nparray, int, int
    """
    layer_fastidxs = fast_distinct[i : i + n_lines]
    fast_line_space = abs(fast_distinct[1] - fast_distinct[0])
    layer_slowidxs = slow_distinct[j : j + n_lines]
    slow_line_space = abs(slow_distinct[0] - slow_distinct[1])
    sample_size = len(segyfile.samples)
    sample_chunck_count = math.ceil(sample_size / n_lines)
    layer_fastmax = max(layer_fastidxs)
    layer_fastmin = min(layer_fastidxs)
    layer_slowmax = max(layer_slowidxs)
    layer_slowmin = min(layer_slowidxs)
    layer_trace_ids = trace_headers[
        (trace_headers.fast >= layer_fastmin)
        & (trace_headers.fast <= layer_fastmax)
        & (trace_headers.slow >= layer_slowmin)
        & (trace_headers.slow <= layer_slowmax)
    ]

    thiscolumn = np.zeros((n_lines, n_lines, sample_chunck_count * n_lines), dtype=np.float32)
    for _, row in layer_trace_ids.iterrows():
        thiscolumn[
            (row[FAST] - layer_fastmin) // fast_line_space,
            (row[SLOW] - layer_slowmin) // slow_line_space,
            0:sample_size,
        ] = segyfile.trace[row.name]

    return thiscolumn, layer_fastmin, layer_slowmin


def _generate_column_blocks(segy_file, n_points, stride, i, j, fast_indexes, slow_indexes, trace_headers):
    """
    Generate arrays for an open segy file (via segyio)
    :param segyio.file segy_file: input segy file previously opened using segyio
    :param int n_points: number of voxels to extract in each dimension
    :param int stride: overlap for output cubes
    :param int i: fast index anchor for origin of column
    :param int j: slow index anchor for origin of column
    :param list fast_indexes: list of distinct fast headers
    :param list slow_indexes: list of distinct slow headers
    :param DataFrame trace_headers: trace headers including fast and slow indexes
    :returns: thiscolumn, fast_anchor, slow_anchor, k
    :rtype: nparray, int, int, int
    """

    sample_size = len(segy_file.samples)
    thiscolumn, fast_anchor, slow_anchor = _get_trace_column(
        n_points, i, j, trace_headers, fast_indexes, slow_indexes, segy_file
    )
    for k in range(0, sample_size - stride, stride):
        yield thiscolumn[i : (i + n_points), j : (j + n_points), k : (k + n_points)], fast_anchor, slow_anchor, k


def _generate_all_blocks(segy_file, n_points, stride, fast_indexes, slow_indexes, trace_headers):
    """
    Generate arrays for an open segy file (via segyio)
    :param segyio.file segy_file: input segy file previously opened using segyio
    :param int n_points: number of voxels to extract in each dimension
    :param int stride: overlap for output cubes
    :param list fast_indexes: list of distinct fast headers
    :param list slow_indexes: list of distinct slow headers
    :param DataFrame trace_headers: trace headers including fast and slow indexes
    :returns: thiscolumn, fast_anchor, slow_anchor, k
    :rtype: nparray, int, int, int
    """
    slow_size = len(slow_indexes)
    fast_size = len(fast_indexes)
    sample_size = len(segy_file.samples)

    # Handle edge case when stride is larger than slow_size and fast_size
    fast_lim = fast_size
    slow_lim = slow_size
    for i in range(0, fast_lim, stride):
        for j in range(0, slow_lim, stride):
            thiscolumn, fast_anchor, slow_anchor = _get_trace_column(
                n_points, i, j, trace_headers, fast_indexes, slow_indexes, segy_file
            )
            for k in range(0, sample_size, stride):
                yield thiscolumn[:, :, k : (k + n_points)], fast_anchor, slow_anchor, k


def timewrapper(func, *args, **kwargs):
    """
    utility function to pass argumentswhile using the timer module
    :param function func: function to wrap
    :param args: parameters accepted by func
    :param kwargs: optional parameters accepted by func
    :returns: wrapped
    :rtype: function
    """

    def wrapped():
        """
        Wrapper function that takes no arguments
        :returns: func
        :rtype: function
        """
        return func(*args, **kwargs)

    return wrapped
