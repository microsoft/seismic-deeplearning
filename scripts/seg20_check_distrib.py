# checks distribution across classes in the new SEG20 competition

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Custom one-off script to process the SEG20 competition test dataset.
"""
import collections

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
TRAIN = "/data/seismic_orig/TrainingData_Labels.segy"
TEST1 = "/home/maxkaz/Desktop/pred_simple_avg_split_test1.segy"
TEST2 = "/home/maxkaz/Desktop/pred_simple_avg_split_test2.segy"

def check(infile):

    data, _ = read_segy(infile)
    n = data.size
    counts = collections.Counter(data.astype(int).flatten().tolist())
    ccounts = 0
    for k in range(1,N_CLASSES+1):
        ccounts += counts[k]
        if k in counts:
            print(f"{k}: {float(counts[k])/n} = {counts[k]} / {n}")
    print(f"coverage {ccounts/n}")

check(TRAIN)
check(TEST1)
check(TEST2)

logging.info("done")
