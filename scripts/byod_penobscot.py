# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Run example:
python byod_penobscot.py --filename <input HDF5 file> --outdir <where to output data>
python prepare_dutchf3.py split_train_val patch   --data_dir=<outdir from the previous step> --label_file=train/train_labels.npy --output_dir=splits --stride=50 --patch_size=100 --split_direction=both --section_stride=100

"""
import sklearn

""" libraries """
import h5py

import numpy as np
import os

np.set_printoptions(linewidth=200)
import logging

# toggle to WARNING when running in production, or use CLI
logging.getLogger().setLevel(logging.DEBUG)~/da
# logging.getLogger().setLevel(logging.WARNING)
import argparse

parser = argparse.ArgumentParser()

""" useful information when running from a GIT folder."""
myname = os.path.realpath(__file__)
mypath = os.path.dirname(myname)
myname = os.path.basename(myname)


def main(args):
    """
    Transforms Penobscot HDF5 dataset into DeepSeismic Tensor Format
    """

    logging.info("loading data")
    f = h5py.File(args.filename, "r")
    data = f["features"][:, :, :, 0]
    labels = f["label"][:, :, :]
    assert labels.min() == 0
    n_classes = labels.max() + 1
    assert n_classes == N_CLASSES

    # inline x depth x crossline, make it inline x crossline x depth
    data = np.swapaxes(data, 1, 2)
    labels = np.swapaxes(labels, 1, 2)

    # Make data cube fast to access
    data = np.ascontiguousarray(data, "float32")
    labels = np.ascontiguousarray(labels, "uint8")

    # combine classes 4 and 5 (index 3 and 4)- shift others down
    labels[labels > 3] -= 1

    # rescale to be within a certain range
    range_min, range_max = -1.0, 1.0
    data_std = (data - data.min()) / (data.max() - data.min())
    data = data_std * (range_max - range_min) + range_min

    """
    # cut off a buffer zone around the volume (to avoid mislabeled data):
    buffer = 25
    data = data[:, buffer:-buffer, buffer:-buffer]
    labels = labels[:, buffer:-buffer, buffer:-buffer]
    """

    # time by crosslines by inlines
    n_inlines = data.shape[0]
    n_crosslines = data.shape[1]

    inline_cut = int(np.floor(n_inlines * INLINE_FRACTION))
    crossline_cut = int(np.floor(n_crosslines * CROSSLINE_FRACTION))

    data_train = data[0:inline_cut, 0:crossline_cut, :]
    data_test1 = data[inline_cut:n_inlines, :, :]
    data_test2 = data[:, crossline_cut:n_crosslines, :]

    labels_train = labels[0:inline_cut, 0:crossline_cut, :]
    labels_test1 = labels[inline_cut:n_inlines, :, :]
    labels_test2 = labels[:, crossline_cut:n_crosslines, :]

    def mkdir(dirname):

        if os.path.isdir(dirname) and os.path.exists(dirname):
            return

        if not os.path.isdir(dirname) and os.path.exists(dirname):
            logging.info("remote file", dirname, "and run this script again")

        os.mkdir(dirname)

    mkdir(args.outdir)
    mkdir(os.path.join(args.outdir, "splits"))
    mkdir(os.path.join(args.outdir, "train"))
    mkdir(os.path.join(args.outdir, "test_once"))

    np.save(os.path.join(args.outdir, "train", "train_seismic.npy"), data_train)
    np.save(os.path.join(args.outdir, "train", "train_labels.npy"), labels_train)

    np.save(os.path.join(args.outdir, "test_once", "test1_seismic.npy"), data_test1)
    np.save(os.path.join(args.outdir, "test_once", "test1_labels.npy"), labels_test1)

    np.save(os.path.join(args.outdir, "test_once", "test2_seismic.npy"), data_test2)
    np.save(os.path.join(args.outdir, "test_once", "test2_labels.npy"), labels_test2)

    # Compute class weights:
    num_classes, class_count = np.unique(labels[:], return_counts=True)
    # class_probabilities = np.histogram(labels[:], bins= , density=True)
    class_weights = 1 - class_count / np.sum(class_count)
    logging.info("CLASS WEIGHTS TO USE")
    logging.info(class_weights)
    logging.info("MEAN")
    logging.info(data.mean())
    logging.info("STANDARD DEVIATION")
    logging.info(data.std())


""" GLOBAL VARIABLES """
INLINE_FRACTION = 0.7
CROSSLINE_FRACTION = 0.78
N_CLASSES = 8

parser.add_argument("--filename", help="Name of HDF5 data", type=str, required=True)
parser.add_argument("--outdir", help="Output data directory location", type=str, required=True)

""" main wrapper with profiler """
if __name__ == "__main__":
    main(parser.parse_args())

# pretty printing of the stack
"""
  try:
    logging.info('before main')
    main(parser.parse_args())
    logging.info('after main')
  except:
    for frame in traceback.extract_tb(sys.exc_info()[2]):
      fname,lineno,fn,text = frame
      print ("Error in %s on line %d" % (fname, lineno))
"""
# optionally enable profiling information
#  import cProfile
#  name = <insert_name_here>
#  cProfile.run('main.run()', name + '.prof')
#  import pstats
#  p = pstats.Stats(name + '.prof')
#  p.sort_stats('cumulative').print_stats(10)
#  p.sort_stats('time').print_stats()
