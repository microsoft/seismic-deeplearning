#!/usr/bin/env python3
""" Please see the def main() function for code description."""
import json

""" libraries """

import numpy as np
import sys
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


def main(args):
    """

    Check to see whether performance metrics are within range on both validation 
    and test sets.

    """
  
    logging.info("loading data")

    with open(args.infile, 'r') as fp:
        data = json.load(fp)

    if args.test:
        # process training set results
        assert data["Pixel Acc: "] > 0.0
        assert data["Pixel Acc: "] <= 1.0
        # TODO make these into proper tests
        # assert data["Pixel Acc: "] == 1.0
        # TODO: add more tests as we fix performance
        # assert data["Mean Class Acc: "] == 1.0
        # assert data["Freq Weighted IoU: "] == 1.0
        # assert data["Mean IoU: "] == 1.0

    else:
        # process validation results
        assert data['Pixelwise Accuracy :'] > 0.0
        assert data['Pixelwise Accuracy :'] <= 1.0
        # TODO make these into proper tests
        # assert data['Pixelwise Accuracy :'] == 1.0
        # TODO: add more tests as we fix performance
        # assert data['Avg loss :'] < 1e-3


    logging.info("all done")


""" GLOBAL VARIABLES """


""" cmd-line arguments """
parser.add_argument("--infile", help="Location of the file which has the metrics", type=str, required=True)
parser.add_argument(
    "--test",
    help="Flag to indicate that these are test set results - validation by default",
    action="store_true"
)

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
