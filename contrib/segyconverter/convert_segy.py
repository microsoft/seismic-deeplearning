# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Utility Script to convert segy files to blocks of numpy arrays and save to individual npy files
"""

import os
import timeit
import argparse
import numpy as np
import utils.segyextract as segyextract
import utils.dataprep as dataprep
import json

K = 12
MIN_VAL = 0
MAX_VAL = 1

def filter_data(output_dir, stddev_file, k, min_range, max_range, clip, normalize):
    """
    Normalization step on all files in output_dir. This function overwrites the existing
    data file
    :param str output_dir: Directory path of all npy files to normalize
    :param str stddev_file: txt file containing standard deviation result
    :param int k: number of standard deviation to be used in normalization
    :param float min_range: minium range value
    :param float max_range: maximum range value
    :param clip: flag to turn on/off clip
    :param normalize: flag to turn on/off normalization.
    """
    txt_file = os.path.join(output_dir, stddev_file)
    if not os.path.isfile(txt_file):
        raise Exception("Std Deviation file could not be found")
    with open(os.path.join(txt_file), 'r') as f:
        metadatastr = f.read()

    try:
        metadata = json.loads(metadatastr)
        stddev = float(metadata['stddev'])
        mean = float(metadata['mean'])
    except ValueError:
        raise Exception('stddev value not valid: {}'.format(metadatastr))

    npy_files = list(f for f in os.listdir(output_dir) if f.endswith('.npy'))
    for local_filename in npy_files:
        cube = np.load(os.path.join(output_dir, local_filename))
        if normalize or clip:
            cube = dataprep.apply(cube, stddev, mean, k, min_range, max_range, 
                                       clip=clip, normalize=normalize)
        np.save(os.path.join(output_dir, local_filename), cube)


def main(input_file, output_dir, prefix, iline=189, xline=193, metadata_only=False, stride=128, 
         cube_size=-1, normalize=True, clip=True):
    """
    Select a single column out of the segy file and generate all cubes in the z(time)
    direction. The column is indexed by the inline and xline. To use this command, you
    should have already run the metadata extract to determine the
    ranges of the inlines and xlines. It will error out if the range is incorrect

    Sample call: python3 convert_segy.py --input_file
                seismic_data.segy --prefix seismic --output_dir ./seismic

    :param str input_file: input segy file path
    :param str output_dir: output directory to save npy files
    :param str prefix: file prefix for npy files
    :param int iline: byte location for inlines
    :param int xline: byte location for crosslines
    :param bool metadata_only: Only return the metadata of the segy file
    :param int stride: overlap between cubes - stride == cube_size = no overlap
    :param int cube_size: size of cubes to generate
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fast_indexes, slow_indexes, trace_headers, sample_size = segyextract.get_segy_metadata(
        input_file, iline, xline)

    print("\tFast Lines: {} to {} ({} lines)".format(np.min(fast_indexes),
          np.max(fast_indexes), len(fast_indexes)))
    print("\tSlow Lines: {} to {} ({} lines)".format(np.min(slow_indexes),
          np.max(slow_indexes), len(slow_indexes)))
    print("\tSample Size: {}".format(sample_size))
    print("\tTrace Count: {}".format(len(trace_headers)))
    print("\tFirst five distinct Fast Line Indexes: {}".format(fast_indexes[0:5]))
    print("\tFirst five distinct Slow Line Indexes: {}".format(slow_indexes[0:5]))
    print("\tFirst five fast trace ids: {}".format(trace_headers['fast'][0:5].values))
    print("\tFirst five slow trace ids: {}".format(trace_headers['slow'][0:5].values))

    if not metadata_only:
        process_time_segy = 0
        if cube_size == -1:
            # only generate on npy
            wrapped_processor_segy = segyextract.timewrapper(segyextract.process_segy_data_into_single_array,
                                                             input_file, output_dir, prefix, iline, xline)
            process_time_segy = timeit.timeit(wrapped_processor_segy, number=1)
        else:
            wrapped_processor_segy = segyextract.timewrapper(segyextract.process_segy_data, input_file,
                                                             output_dir, prefix, stride=stride, n_points=cube_size)
            process_time_segy = timeit.timeit(wrapped_processor_segy, number=1)
        print(f"Completed SEG-Y converstion in: {process_time_segy}")
        # At this point, there should be npy files in the output directory + one file containing the std deviation found in the segy
        print("Preparing File")
        timed_filter_data = segyextract.timewrapper(filter_data, output_dir, f"{prefix}_stats.json",
                                                    K, MIN_VAL, MAX_VAL, clip=clip, normalize=normalize)
        process_time_normalize = timeit.timeit(timed_filter_data, number=1)
        print(f"Completed file preparation in {process_time_normalize} seconds")


if __name__ == '__main__':

    parser = argparse.ArgumentParser("train")
    parser.add_argument("--prefix", type=str, help="prefix label for output files",
                        required=True)
    parser.add_argument("--input_file", type=str, help="segy file path", required=True)
    parser.add_argument("--output_dir", type=str,
                        help="Output files are written to this directory", default='.')
    parser.add_argument("--metadata_only", action='store_true',
                        help="Only produce inline,xline metadata")
    parser.add_argument("--iline", type=int, default=189,
                        help="segy file path")
    parser.add_argument("--xline", type=int, default=193,
                        help="segy file path")
    parser.add_argument("--cube_size", type=int, default=-1,
                        help="cube dimensions")
    parser.add_argument("--stride", type=int, default=128,
                        help="stride")
    parser.add_argument("--normalize", action='store_true', help="Normalization flag -  clip and normalize the data")
    parser.add_argument("--clip", action='store_true', help="Clipping flag -  only clip the data")

    args = parser.parse_args()
    localfile = args.input_file

    main(args.input_file, args.output_dir, args.prefix, args.iline, args.xline,
         args.metadata_only, args.stride, args.cube_size, args.normalize, args.clip)
