# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Custom one-off script to process the SEG20 competition test dataset.
"""

from deepseismic_interpretation.data import read_segy

""" libraries """
import segyio

import numpy as np
from scipy import stats
import os

np.set_printoptions(linewidth=200)
import logging

# toggle to WARNING when running in production, or use CLI
logging.getLogger().setLevel(logging.DEBUG)

# dataset locations
N_CLASSES = 6
TEST1 = "/data/seismic/TestData_Image1.segy"
TEST2 = "/data/seismic/TestData_Image2.segy"
# output location
OUTDIR = "/data/seismic/test_once"
# enter these from byod_competition logging output - computed on the training set
MEAN = 0.676609992980957
STD = 390.308837890625
MIN = -1170.2498779296875
MAX = 1171.6031494140625

def process_test(infile, outdir, n_set):

    logging.info("loading data")
    data, _ = read_segy(infile)

    logging.info("Running 3-sigma clipping")
    clip_scaling = 3.0
    data[data > MEAN + clip_scaling * STD] = MEAN + clip_scaling * STD
    data[data < MEAN - clip_scaling * STD] = MEAN - clip_scaling * STD

    # Make data cube fast to access
    logging.info("Adjusting precision")
    data = np.ascontiguousarray(data, "float32")

    # rescale to be within a certain range
    range_min, range_max = -1.0, 1.0
    data_std = (data - MIN) / (MAX - MIN)
    data = data_std * (range_max - range_min) + range_min

    random_test_labels = np.random.randint(0,N_CLASSES-1, data.shape, dtype='uint8')
    np.save(os.path.join(outdir, f"test{n_set}_seismic.npy"), data)
    np.save(os.path.join(outdir, f"test{n_set}_labels.npy"), random_test_labels)

process_test(TEST1, OUTDIR, 1)
process_test(TEST2, OUTDIR, 2)
