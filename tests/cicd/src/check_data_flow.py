#!/usr/bin/env python3
""" Please see the def main() function for code description."""
import json


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


def main(args):
    """

    Tests to ensure proper data flow throughout the experiments.

    """

    logging.info("loading data")

    with open(args.infile, "r") as fp:
        data = json.load(fp)

    # Note: these are specific to the setup in 
    # main_build.yml for train.py 
    # and get_data_for_builds.sh and prepare_dutchf3.py and prepare_dutchf3.py

    if args.step=='test':

        for test_key in data.keys():
            if args.train_depth=='none':
                assert tuple(data[test_key]['test_input_shape'])==(200, 200, 200) 
                for img in data[test_key]["img_shape"]:
                    assert tuple(img)==(1, 1, 200, 200) #dimenstion
            
            elif args.train_depth=='section':
                assert tuple(data[test_key]['test_input_shape'])==(200, 3, 200, 200) 
                for img in data[test_key]["img_shape"]:
                    assert tuple(img)==(1, 3, 200, 200) #dimenstion
            
            elif args.train_depth=='patch':
                raise Exception('Must be added')
    
            
            assert len(data[test_key]["pred_shape"])==data[test_key]["take_n_sections"]
            assert len(data[test_key]["gt_shape"])==data[test_key]["take_n_sections"]
            assert len(data[test_key]["img_shape"])==data[test_key]["take_n_sections"]

            assert tuple(data[test_key]["test_label_shape"])==(200, 200, 200)

            assert data[test_key]["test_section_loader_length"]==400
            assert data[test_key]["n_classes"]==2
            assert data[test_key]["test_loader_length"]==400

            for pred, gt in zip(data[test_key]["pred_shape"], data[test_key]["gt_shape"]):
                assert tuple(pred[0])==(1, 200, 200) #dimenstion
                assert pred[1]<=2 # unique classes

                assert tuple(gt[0])==(1, 1, 200, 200) #dimenstion
                assert gt[1]<=2 # unique classes
        


    elif args.step=='train':
        if args.train_depth=='none':
            assert tuple(data['train_input_shape'])==(200, 200, 200)
        elif args.train_depth=='section':
            assert tuple(data['train_input_shape'])==(200, 3, 200, 200)
        elif args.train_depth=='patch':
            raise Exception('Must be added')
        
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

