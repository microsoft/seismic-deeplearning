# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Utility functions for pytest
"""
import numpy as np
import os
import pytest
import segyio





def is_npy(s):
    """
    Filter check for npy files
    :param str s: file path
    :returns: True if npy
    :rtype: bool
    """
    if (s.find(".npy") == -1):
        return False
    else:
        return True


def get_npy_files(outputdir):
    """
    List npy files
    :param str outputdir: location of npy files
    :returns: npy_files
    :rtype: list
    """
    npy_files = os.listdir(outputdir)
    npy_files = list(filter(is_npy, npy_files))
    npy_files.sort()
    return npy_files


def build_volume(n_points, npy_files, file_location):
    """
    Rebuild volume from npy files. This only works for a vertical column of
    npy files. If there is a cube of files, then a new algorithm will be required to
    stitch them back together

    :param int n_points: size of cube expected in npy_files
    :param list npy_files: list of files to load into vertical volume
    :param str file_location: directory for npy files to add to array
    :returns: numpy array created by stacking the npy_file arrays vertically (third axis)
    :rtype: numpy.array
    """
    full_volume_from_file = np.zeros((n_points, n_points, n_points * len(npy_files)), dtype=np.float32)
    for i, file in enumerate(npy_files):
        data = np.load(os.path.join(file_location, file))
        full_volume_from_file[:, :, n_points * i:n_points * (i + 1)] = data
    return full_volume_from_file


def create_segy_file(masklambda, filename, sorting=segyio.TraceSortingFormat.INLINE_SORTING, ilinerange=[10, 50], xlinerange=[100, 300]):
    
    # segyio.spec is the minimum set of values for a valid segy file.
    spec = segyio.spec()
    spec.sorting = 2
    spec.format = 1
    spec.samples = range(int(10))
    spec.ilines = range(*map(int, ilinerange))
    spec.xlines = range(*map(int, xlinerange))
    print(f"Written to {filename}")
    print(f"\tinlines: {len(spec.ilines)}")
    print(f"\tcrosslines: {len(spec.xlines)}")

    with segyio.create(filename, spec) as f:
        # one inline consists of 50 traces
        # which in turn consists of 2000 samples
        step = 0.00001
        start = step * len(spec.samples)
        # fill a trace with predictable values: left-of-comma is the inline
        # number. Immediately right of comma is the crossline number
        # the rightmost digits is the index of the sample in that trace meaning
        # looking up an inline's i's jth crosslines' k should be roughly equal
        # to i.j0k
        trace = np.linspace(-1,1,len(spec.samples),True,dtype=np.single)
        if sorting == segyio.TraceSortingFormat.INLINE_SORTING:
            # Write the file trace-by-trace and update headers with iline, xline
            # and offset
            tr = 0
            for il in spec.ilines:
                for xl in spec.xlines:
                    if masklambda(il, xl):
                        f.header[tr] = {
                            segyio.su.offset: 1,
                            segyio.su.iline: il,
                            segyio.su.xline: xl
                        }
                        f.trace[tr] = trace * ((xl / 100.0) + il)
                        tr += 1
                      
            f.bin.update(
                tsort=segyio.TraceSortingFormat.INLINE_SORTING
            )
        else:
            # Write the file trace-by-trace and update headers with iline, xline
            # and offset
            tr = 0
            for il in spec.ilines:
                for xl in spec.xlines:
                    if masklambda(il, xl):
                        f.header[tr] = {
                            segyio.su.offset: 1,
                            segyio.su.iline: il,
                            segyio.su.xline: xl
                        }
                        f.trace[tr] = trace * (xl / 100.0) + il
                        tr += 1
                 
            f.bin.update(
                tsort=segyio.TraceSortingFormat.CROSSLINE_SORTING
            )
        # Add some noise for clipping and normalization tests
        f.trace[tr // 2] = trace * ((max(spec.xlines) / 100.0) + max(spec.ilines)) * 20
        f.trace[tr // 3] = trace * ((min(spec.xlines) / 100.0) + min(spec.ilines)) * 20
        print(f"\ttraces: {tr}")