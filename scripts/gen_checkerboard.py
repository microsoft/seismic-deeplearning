#!/usr/bin/env python3
""" Please see the def main() function for code description."""

""" libraries """

import numpy as np
import sys
import os
np.set_printoptions(linewidth=200)
import logging
# toggle to WARNING when running in production, or use CLI
logging.getLogger().setLevel(logging.DEBUG)
#logging.getLogger().setLevel(logging.WARNING)
import argparse
parser = argparse.ArgumentParser()

""" useful information when running from a GIT folder."""
myname = os.path.realpath(__file__)
mypath = os.path.dirname(myname)
myname = os.path.basename(myname)


def make_box(n_inlines, n_crosslines, n_depth, box_size):
    """
    Makes a 3D box in checkerboard pattern.

    :param n_inlines: dim x
    :param n_crosslines: dim y
    :param n_depth: dim z
    :return: numpy array
    """
    # inline by crossline by depth
    zero_patch = np.ones(box_size) * WHITE
    one_patch = np.ones(box_size) * BLACK

    # multiplying by one flips the sign
    stride = np.hstrack((zero_patch, one_patch))
    neg_stride = -1 * stride

    # come up with a 2D inline image
    nx, ny = stride.shape

    step_col = int(np.ceil(n_crosslines / float(ny)))
    step_row = int(np.ceil(n_inlines / float(nx) / 2))

    # move in the crossline direction
    crossline_band = np.hstack((stride,) * step_col)
    neg_crossline_band = -1 * crossline_band

    checker_stripe = np.vstack((crossline_band, neg_crossline_band))

    # move down, alternative
    checker_image = np.vstack((checker_stripe,) * step_row)

    # trim excess
    checker_image = checker_image[0:n_inlines, 0:n_crosslines]

    # now make a box with alternating checkers
    checker_box = np.ones((n_inlines, n_crosslines, box_size * 2))
    checker_box[:, :, 0:box_size] = checker_image
    # now invert the colors
    checker_box[:, :, box_size:] = -1 * checker_image

    # stack boxes depth wise
    step_depth = int(np.ceil(n_depth / float(box_size) / 2))
    final_box = np.concatenate((checker_box,) * step_depth, axis=2)


def main(args):
    """

    TODO

    TODO: please see the TODO statements in the code

    """

    logging.info('loading data')

    train_seismic = np.load(os.path.join(args.dataroot, "train", "train_seismic.npy"))
    train_labels = np.load(os.path.join(args.dataroot, "train", "train_labels.npy"))
    test1_seismic = np.load(path.join(data_dir, "test_once", "test1_seismic.npy"))
    test1_labels = np.load(path.join(data_dir, "test_once", "test1_labels.npy"))
    test2_seismic = np.load(path.join(data_dir, "test_once", "test2_seismic.npy"))
    test2_labels = np.load(path.join(data_dir, "test_once", "test2_labels.npy"))

    assert train_seismic.shape == train_labels.shape
    assert train_seismic.min() == WHITE
    assert train_seismic.max() == BLACK

    n_inlines, n_crosslines, n_depth = train_seismic.shape

    logging.info('train checkerbox')
    checkerboard_train_seismic = make_box(n_inlines, n_crosslines, n_depth, args.box_size)
    checkerboard_train_seismic = checkerboard_train_seismic.astype(train_seismic.dtype)
    checkerboard_train_labels = checkerboard_train_seismic.astype(train_labels.dtype)

    # create checkerbox
    logging.info('test1 checkerbox')
    n_inlines, n_crosslines, n_depth = train_seismic.shape
    checkerboard_train_seismic = make_box(n_inlines, n_crosslines, n_depth, args.box_size)
    checkerboard_train_seismic = checkerboard_train_seismic.astype(train_seismic.dtype)
    checkerboard_train_labels = checkerboard_train_seismic.astype(train_labels.dtype)

    logging.info('test2 checkerbox')
    n_inlines, n_crosslines, n_depth = train_seismic.shape
    checkerboard_train_seismic = make_box(n_inlines, n_crosslines, n_depth, args.box_size)
    checkerboard_train_seismic = checkerboard_train_seismic.astype(train_seismic.dtype)
    checkerboard_train_labels = checkerboard_train_seismic.astype(train_labels.dtype)
   
    logging.info('all done')

""" GLOBAL VARIABLES """
WHITE = -1
BLACK = 1

parser.add_argument("--dataroot", help="Root location of the input data", type=str, required = True, default = None)
parser.add_argument("--dataout", help="Root location of the output data", type=str, required = True, default = None)
parser.add_argument("--box_size", help="Size of the bounding box", type=int, required = False, default = 100)

""" main wrapper with profiler """
if __name__ == "__main__":
  main(parser.parse_args())
  
# pretty printing of the stack
'''
  try:
    logging.info('before main')
    main(parser.parse_args())
    logging.info('after main')
  except:
    for frame in traceback.extract_tb(sys.exc_info()[2]):
      fname,lineno,fn,text = frame
      print ("Error in %s on line %d" % (fname, lineno))
'''
# optionally enable profiling information
#  import cProfile
#  name = <insert_name_here>
#  cProfile.run('main.run()', name + '.prof')
#  import pstats
#  p = pstats.Stats(name + '.prof')
#  p.sort_stats('cumulative').print_stats(10)
#  p.sort_stats('time').print_stats()
