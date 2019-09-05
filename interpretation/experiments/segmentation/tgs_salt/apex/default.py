# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ""
_C.LOG_DIR = ""
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.LOG_CONFIG = "logging.conf"

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = "/mnt/tgssalt"
_C.DATASET.NUM_CLASSES = 1

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = "seg_hrnet"
_C.MODEL.PRETRAINED = ""
_C.MODEL.EXTRA = CN(new_allowed=True)

_C.LOSS = CN()
_C.LOSS.WEIGHTS = (0.01, 1)
_C.LOSS.ADJUST_EPOCH = 50
_C.LOSS.ADJUSTED_WEIGHTS = (0.3, 0.7)


# training
_C.TRAIN = CN()
_C.TRAIN.MIN_LR = 0.001
_C.TRAIN.MAX_LR = 0.01
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 484
_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.WEIGHT_DECAY = 0.0001
_C.TRAIN.PAD_LEFT = 27
_C.TRAIN.PAD_RIGHT = 27
_C.TRAIN.FINE_SIZE = 202
_C.TRAIN.SNAPSHOTS = 5
_C.TRAIN.SAVE_LOCATION = "/tmp/models"

# testing
_C.TEST = CN()
_C.TEST.BATCH_SIZE_PER_GPU = 32
_C.TEST.CV = CN()
_C.TEST.CV.N_SPLITS = 5
_C.TEST.CV.SEED = 42
_C.TEST.CV.SHUFFLE = True


def update_config(cfg, options=None, config_file=None):
    cfg.defrost()

    if config_file:
        cfg.merge_from_file(config_file)

    if options:
        cfg.merge_from_list(options)

    cfg.freeze()


if __name__ == "__main__":
    import sys

    with open(sys.argv[1], "w") as f:
        print(_C, file=f)

