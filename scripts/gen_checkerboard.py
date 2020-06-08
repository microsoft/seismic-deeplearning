#!/usr/bin/env python3
""" Please see the def main() function for code description."""

""" libraries """

import numpy as np
import os

np.set_printoptions(linewidth=200)
import logging

# toggle to WARNING when running in production, or use CLI
logging.getLogger().setLevel(logging.DEBUG)
# logging.getLogger().setLevel(logging.WARNING)
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
    :param box_size: size of each checkerbox
    :return: numpy array
    """
    # inline by crossline by depth
    zero_patch = np.ones((box_size, box_size)) * WHITE
    one_patch = np.ones((box_size, box_size)) * BLACK

    stride = np.hstack((zero_patch, one_patch))

    # come up with a 2D inline image
    nx, ny = stride.shape

    step_col = int(np.ceil(n_crosslines / float(ny)))
    step_row = int(np.ceil(n_inlines / float(nx) / 2))

    # move in the crossline direction
    crossline_band = np.hstack((stride,) * step_col)
    # multiplying by negative one flips the sign
    neg_crossline_band = -1 * crossline_band

    checker_stripe = np.vstack((crossline_band, neg_crossline_band))

    # move down a section
    checker_image = np.vstack((checker_stripe,) * step_row)

    # trim excess
    checker_image = checker_image[0:n_inlines, 0:n_crosslines]

    # now make a box with alternating checkers
    checker_box = np.ones((n_inlines, n_crosslines, box_size * 2))
    checker_box[:, :, 0:box_size] = checker_image[:, :, np.newaxis]
    # now invert the colors
    checker_box[:, :, box_size:] = -1 * checker_image[:, :, np.newaxis]

    # stack boxes depth wise
    step_depth = int(np.ceil(n_depth / float(box_size) / 2))
    final_box = np.concatenate((checker_box,) * step_depth, axis=2)

    # trim excess again
    return final_box[0:n_inlines, 0:n_crosslines, 0:n_depth]


def make_gradient(n_inlines, n_crosslines, n_depth, box_size, dir="inline"):
    """
    Makes a 3D box gradient pattern in a particular direction

    :param n_inlines: dim x
    :param n_crosslines: dim y
    :param n_depth: dim z
    :param box_size: thichkness of gradient box
    :param dir: direction of the gradient - can be crossline, inline or depth
    :return: numpy array
    """

    orthogonal_dir = dir # for depth case
    if dir=='inline':
        orthogonal_dir = 'crossline'
    elif dir=='crossline':
        orthogonal_dir = 'inline'
    
    axis = GRADIENT_DIR.index(orthogonal_dir)
    
    n_points = (n_inlines, n_crosslines, n_depth)[axis]
    n_classes = int(np.ceil(float(n_points) / box_size))
    logging.info(f"GRADIENT: we will output {n_classes} classes in the {dir} direction")

    output = np.ones((n_inlines, n_crosslines, n_depth))
    start, finish = 0, box_size
    for i in range(n_classes):
        sl = [slice(None)] * output.ndim
        sl[axis] = range(start, finish)
        # set all values equal to class number, starting from 0
        output[tuple(sl)] = i
        start += box_size
        finish = min(n_points, finish + box_size)

    return output


def mkdir(path):
    """
  Create a directory helper function
  """
    if not os.path.isdir(path):
        os.mkdir(path)


def main(args):
    """

    Generates checkerboard dataset based on Dutch F3 in Alaudah format.

    Pre-requisite: valid Dutch F3 dataset in Alaudah format.

    """

    logging.info("loading data")

    train_seismic = np.load(os.path.join(args.dataroot, "train", "train_seismic.npy"))
    train_labels = np.load(os.path.join(args.dataroot, "train", "train_labels.npy"))
    test1_seismic = np.load(os.path.join(args.dataroot, "test_once", "test1_seismic.npy"))
    test1_labels = np.load(os.path.join(args.dataroot, "test_once", "test1_labels.npy"))
    test2_seismic = np.load(os.path.join(args.dataroot, "test_once", "test2_seismic.npy"))
    test2_labels = np.load(os.path.join(args.dataroot, "test_once", "test2_labels.npy"))

    assert train_seismic.shape == train_labels.shape
    assert train_seismic.min() == WHITE
    assert train_seismic.max() == BLACK
    assert train_labels.min() == 0
    # this is the number of classes in Alaudah's Dutch F3 dataset
    assert train_labels.max() == 5

    assert test1_seismic.shape == test1_labels.shape
    assert test1_seismic.min() == WHITE
    assert test1_seismic.max() == BLACK
    assert test1_labels.min() == 0
    # this is the number of classes in Alaudah's Dutch F3 dataset
    assert test1_labels.max() == 5

    assert test2_seismic.shape == test2_labels.shape
    assert test2_seismic.min() == WHITE
    assert test2_seismic.max() == BLACK
    assert test2_labels.min() == 0
    # this is the number of classes in Alaudah's Dutch F3 dataset
    assert test2_labels.max() == 5

    if args.type == "checkerboard":
        logging.info("train checkerbox")
        n_inlines, n_crosslines, n_depth = train_seismic.shape
        checkerboard_train_seismic = make_box(n_inlines, n_crosslines, n_depth, args.box_size)
        checkerboard_train_seismic = checkerboard_train_seismic.astype(train_seismic.dtype)
        checkerboard_train_labels = checkerboard_train_seismic.astype(train_labels.dtype)
        # labels are integers and start from zero
        checkerboard_train_labels[checkerboard_train_seismic < WHITE_LABEL] = WHITE_LABEL

        # create checkerbox
        logging.info("test1 checkerbox")
        n_inlines, n_crosslines, n_depth = test1_seismic.shape
        checkerboard_test1_seismic = make_box(n_inlines, n_crosslines, n_depth, args.box_size)
        checkerboard_test1_seismic = checkerboard_test1_seismic.astype(test1_seismic.dtype)
        checkerboard_test1_labels = checkerboard_test1_seismic.astype(test1_labels.dtype)
        # labels are integers and start from zero
        checkerboard_test1_labels[checkerboard_test1_seismic < WHITE_LABEL] = WHITE_LABEL

        logging.info("test2 checkerbox")
        n_inlines, n_crosslines, n_depth = test2_seismic.shape
        checkerboard_test2_seismic = make_box(n_inlines, n_crosslines, n_depth, args.box_size)
        checkerboard_test2_seismic = checkerboard_test2_seismic.astype(test2_seismic.dtype)
        checkerboard_test2_labels = checkerboard_test2_seismic.astype(test2_labels.dtype)
        # labels are integers and start from zero
        checkerboard_test2_labels[checkerboard_test2_seismic < WHITE_LABEL] = WHITE_LABEL

    # substitute gradient dataset instead of checkerboard
    elif args.type == "gradient":

        logging.info("train gradient")
        n_inlines, n_crosslines, n_depth = train_seismic.shape
        checkerboard_train_seismic = make_gradient(
            n_inlines, n_crosslines, n_depth, args.box_size, dir=args.gradient_dir
        )
        checkerboard_train_seismic = checkerboard_train_seismic.astype(train_seismic.dtype)
        checkerboard_train_labels = checkerboard_train_seismic.astype(train_labels.dtype)
        # labels are integers and start from zero
        checkerboard_train_labels[checkerboard_train_seismic < WHITE_LABEL] = WHITE_LABEL

        # create checkerbox
        logging.info("test1 gradient")
        n_inlines, n_crosslines, n_depth = test1_seismic.shape
        checkerboard_test1_seismic = make_gradient(
            n_inlines, n_crosslines, n_depth, args.box_size, dir=args.gradient_dir
        )
        checkerboard_test1_seismic = checkerboard_test1_seismic.astype(test1_seismic.dtype)
        checkerboard_test1_labels = checkerboard_test1_seismic.astype(test1_labels.dtype)
        # labels are integers and start from zero
        checkerboard_test1_labels[checkerboard_test1_seismic < WHITE_LABEL] = WHITE_LABEL

        logging.info("test2 gradient")
        n_inlines, n_crosslines, n_depth = test2_seismic.shape
        checkerboard_test2_seismic = make_gradient(
            n_inlines, n_crosslines, n_depth, args.box_size, dir=args.gradient_dir
        )
        checkerboard_test2_seismic = checkerboard_test2_seismic.astype(test2_seismic.dtype)
        checkerboard_test2_labels = checkerboard_test2_seismic.astype(test2_labels.dtype)
        # labels are integers and start from zero
        checkerboard_test2_labels[checkerboard_test2_seismic < WHITE_LABEL] = WHITE_LABEL

    # substitute binary dataset instead of checkerboard
    elif args.type == "binary":

        logging.info("train binary")
        checkerboard_train_seismic = train_seismic * 0 + WHITE
        checkerboard_train_labels = train_labels * 0 + WHITE_LABEL

        # create checkerbox
        logging.info("test1 binary")
        checkerboard_test1_seismic = test1_seismic * 0 + BLACK
        checkerboard_test1_labels = test1_labels * 0 + BLACK_LABEL

        logging.info("test2 binary")
        checkerboard_test2_seismic = test2_seismic * 0 + BLACK
        checkerboard_test2_labels = test2_labels * 0 + BLACK_LABEL

    logging.info("writing data to disk")
    mkdir(args.dataout)
    mkdir(os.path.join(args.dataout, "data"))
    mkdir(os.path.join(args.dataout, "data", "splits"))
    mkdir(os.path.join(args.dataout, "data", "train"))
    mkdir(os.path.join(args.dataout, "data", "test_once"))

    np.save(os.path.join(args.dataout, "data", "train", "train_seismic.npy"), checkerboard_train_seismic)
    np.save(os.path.join(args.dataout, "data", "train", "train_labels.npy"), checkerboard_train_labels)

    np.save(os.path.join(args.dataout, "data", "test_once", "test1_seismic.npy"), checkerboard_test1_seismic)
    np.save(os.path.join(args.dataout, "data", "test_once", "test1_labels.npy"), checkerboard_test1_labels)

    np.save(os.path.join(args.dataout, "data", "test_once", "test2_seismic.npy"), checkerboard_test2_seismic)
    np.save(os.path.join(args.dataout, "data", "test_once", "test2_labels.npy"), checkerboard_test2_labels)

    logging.info("all done")


""" GLOBAL VARIABLES """
WHITE = -1
BLACK = 1
WHITE_LABEL = 0
BLACK_LABEL = BLACK
TYPES = ["checkerboard", "gradient", "binary"]
GRADIENT_DIR = ["inline", "crossline", "depth"]

parser.add_argument("--dataroot", help="Root location of the input data", type=str, required=True)
parser.add_argument("--dataout", help="Root location of the output data", type=str, required=True)
parser.add_argument("--box_size", help="Size of the bounding box", type=int, required=False, default=100)
parser.add_argument(
    "--type", help="Type of data to generate", type=str, required=False, choices=TYPES, default="checkerboard",
)
parser.add_argument(
    "--gradient_dir",
    help="Direction in which to build the gradient",
    type=str,
    required=False,
    choices=GRADIENT_DIR,
    default="inline",
)
parser.add_argument("--debug", help="Turn on debug mode", type=bool, required=False, default=False)

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
