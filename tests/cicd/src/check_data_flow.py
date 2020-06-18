#!/usr/bin/env python3
""" Please see the def main() function for code description."""
import json
import math

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

    Tests to ensure proper data flow throughout the experiments.

    """

    logging.info("loading data")

    with open(args.infile, "r") as fp:
        data = json.load(fp)

    

    if args.step=='test':
        pass
    elif args.step=='train':
        if args.train_depth=='none':
            assert tuple(data['train_input_shape'])==(200, 200, 200)
            
        elif args.train_depth=='section':
            assert tuple(data['train_input_shape'])==(200, 3, 200, 200)
        elif args.train_depth=='patch':
            pass
        assert tuple(data['train_label_shape'])==(200, 200, 200)
        assert data['train_patch_loader_length']==64
        assert data['validation_patch_loader_length']==1280
        assert data['train_length_subset']==64
        assert data['validation_length_subset']==32
        assert data['train_loader_length']==4
        assert data['validation_loader_length']==1
        assert data['n_classes']==2
        


    

    logging.info("all done")



""" cmd-line arguments """
STEPS = ["test", "train"]
TRAIN_DEPTH = ["none", "patch", "section"]

parser.add_argument("--infile", help="Location of the file which has the metrics", type=str, required=True)
parser.add_argument("--step", choices=STEPS, type=str, required=True, help="Data flow checks for test or training pipeline")
parser.add_argument("--train_depth", choices=TRAIN_DEPTH, type=str, required=True, help="Train depth flag, to check the dimensions")
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
