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

    if args.step == "test":

        for test_key in data.keys():
            if args.train_depth == "none":
                expected_test_input_shape = (200, 200, 200)
                expected_img = (1, 1, 200, 200)

            elif args.train_depth == "section":
                expected_test_input_shape = (200, 3, 200, 200)
                expected_img = (1, 3, 200, 200)

            elif args.train_depth == "patch":
                expected_test_input_shape = "TBD"
                expected_img = "TBD"
                raise Exception("Must be added")

            msg = f"Expected {expected_test_input_shape} for shape, received {tuple(data[test_key]['test_input_shape'])} instead, in {args.infile.split('.')[0]}"
            assert tuple(data[test_key]["test_input_shape"]) == expected_test_input_shape, msg

            expected_test_label_shape = (200, 200, 200)
            msg = f"Expected {expected_test_label_shape} for shape, received {tuple(data[test_key]['test_label_shape'])} instead, in {args.infile.split('.')[0]}"
            assert tuple(data[test_key]["test_label_shape"]) == expected_test_label_shape, msg

            for img in data[test_key]["img_shape"]:
                msg = (
                    f"Expected {expected_img} for shape, received {tuple(img)} instead, in {args.infile.split('.')[0]}"
                )
                assert tuple(img) == expected_img, msg

            # -----------------------------------------------
            exp_n_section = data[test_key]["take_n_sections"]
            pred_shape_len = len(data[test_key]["pred_shape"])
            msg = f"Expected {exp_n_section} number of items, received {pred_shape_len} instead, in {args.infile.split('.')[0]}"
            assert pred_shape_len == exp_n_section, msg

            gt_shape_len = len(data[test_key]["gt_shape"])
            msg = f"Expected {exp_n_section} number of items, received {gt_shape_len} instead, in {args.infile.split('.')[0]}"
            assert gt_shape_len == exp_n_section, msg

            img_shape_len = len(data[test_key]["img_shape"])
            msg = f"Expected {exp_n_section} number of items, received {img_shape_len} instead, in {args.infile.split('.')[0]}"
            assert img_shape_len == exp_n_section, msg

            expected_len = 400
            lhs_assertion = data[test_key]["test_section_loader_length"]
            msg = f"Expected {expected_len} for test section loader length, received {lhs_assertion} instead, in {args.infile.split('.')[0]}"
            assert lhs_assertion == expected_len, msg

            lhs_assertion = data[test_key]["test_loader_length"]
            msg = f"Expected {expected_len} for test loader length, received {lhs_assertion} instead, in {args.infile.split('.')[0]}"
            assert lhs_assertion == expected_len, msg

            expected_n_classes = 2
            lhs_assertion = data[test_key]["n_classes"]
            msg = f"Expected {expected_n_classes} for test loader length, received {lhs_assertion} instead, in {args.infile.split('.')[0]}"
            assert lhs_assertion == expected_n_classes, msg

            expected_pred = (1, 200, 200)
            expected_gt = (1, 1, 200, 200)

            for pred, gt in zip(data[test_key]["pred_shape"], data[test_key]["gt_shape"]):
                # dimenstion
                msg = f"Expected {expected_pred} for prediction shape, received {tuple(pred[0])} instead, in {args.infile.split('.')[0]}"
                assert tuple(pred[0]) == expected_pred, msg

                # unique classes
                msg = f"Expected up to {expected_n_classes} unique prediction classes, received {pred[1]} instead, in {args.infile.split('.')[0]}"
                assert pred[1] <= expected_n_classes, msg

                # dimenstion
                msg = f"Expected {expected_gt} for ground truth mask shape, received {tuple(gt[0])} instead, in {args.infile.split('.')[0]}"
                assert tuple(gt[0]) == expected_gt, msg

                # unique classes
                msg = f"Expected up to {expected_n_classes} unique ground truth classes, received {gt[1]} instead, in {args.infile.split('.')[0]}"
                assert gt[1] <= expected_n_classes, msg

    elif args.step == "train":
        if args.train_depth == "none":
            expected_shape_in = (200, 200, 400)
        elif args.train_depth == "section":
            expected_shape_in = (200, 3, 200, 400)
        elif args.train_depth == "patch":
            expected_shape_in = "TBD"
            raise Exception("Must be added")

        msg = f"Expected {expected_shape_in} for shape, received {tuple(data['train_input_shape'])} instead, in {args.infile.split('.')[0]}"
        assert tuple(data["train_input_shape"]) == expected_shape_in, msg

        expected_shape_label = (200, 200, 400)
        msg = f"Expected {expected_shape_label} for shape, received {tuple(data['train_label_shape'])} instead, in {args.infile.split('.')[0]}"
        assert tuple(data["train_label_shape"]) == expected_shape_label, msg

        expected_len = 64
        msg = f"Expected {expected_len} for train patch loader length, received {data['train_patch_loader_length']} instead, in {args.infile.split('.')[0]}"
        assert data["train_patch_loader_length"] == expected_len, msg

        expected_len = 1280
        msg = f"Expected {expected_len} for validation patch loader length, received {data['validation_patch_loader_length']} instead, in {args.infile.split('.')[0]}"
        assert data["validation_patch_loader_length"] == expected_len, msg

        expected_len = 64
        msg = f"Expected {expected_len} for train subset length, received {data['train_length_subset']} instead, in {args.infile.split('.')[0]}"
        assert data["train_length_subset"] == expected_len, msg

        expected_len = 32
        msg = f"Expected {expected_len} for validation subset length, received {data['validation_length_subset']} instead, in {args.infile.split('.')[0]}"
        assert data["validation_length_subset"] == expected_len, msg

        expected_len = 4
        msg = f"Expected {expected_len} for train loader length, received {data['train_loader_length']} instead, in {args.infile.split('.')[0]}"
        assert data["train_loader_length"] == expected_len, msg

        expected_len = 1
        msg = f"Expected {expected_len} for train loader length, received {data['train_loader_length']} instead, in {args.infile.split('.')[0]}"
        assert data["validation_loader_length"] == expected_len, msg

        expected_n_classes = 2
        msg = f"Expected {expected_n_classes} for number of classes, received {data['n_classes']} instead, in {args.infile.split('.')[0]}"
        assert data["n_classes"] == expected_n_classes, msg

    logging.info("all done")


""" cmd-line arguments """
STEPS = ["test", "train"]
TRAIN_DEPTH = ["none", "patch", "section"]

parser.add_argument("--infile", help="Location of the file which has the metrics", type=str, required=True)
parser.add_argument(
    "--step", choices=STEPS, type=str, required=True, help="Data flow checks for test or training pipeline"
)
parser.add_argument(
    "--train_depth", choices=TRAIN_DEPTH, type=str, required=True, help="Train depth flag, to check the dimensions"
)
""" main wrapper with profiler """
if __name__ == "__main__":
    main(parser.parse_args())

