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

_C.OUTPUT_DIR = "output"
_C.LOG_DIR = "log"
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.LOG_CONFIG = "/data/home/mat/repos/DeepSeismic/logging.conf"
_C.SEED = 42

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = "/mnt/alaudah"
_C.DATASET.NUM_CLASSES = 1
_C.DATASET.STRIDE = 50
_C.DATASET.PATCH_SIZE = 99
_C.DATASET.AUGMENTATION = True
_C.DATASET.DEPTH = 'no' # Options are None, Patch and Section
# None adds no depth information and the num of channels remains at 1
# Patch adds depth per patch so is simply the height of that patch from 0 to 1, channels=3
# Section adds depth per section so contains depth information for the whole section, channels=3

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = "patch_deconvnet"
_C.MODEL.IN_CHANNELS = 1
_C.MODEL.NUM_CLASSES = 6

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
_C.TRAIN.SNAPSHOTS = 5
_C.TRAIN.SAVE_LOCATION = "/tmp/models"
_C.TRAIN.AUGMENTATION = True

# validation
_C.VALIDATION = CN()
_C.VALIDATION.BATCH_SIZE_PER_GPU = 32

# TEST
_C.TEST = CN()
_C.TEST.MODEL_PATH = ""
_C.TEST.TEST_STRIDE = 10
_C.TEST.SPLIT = 'Both' # Can be Both, Test1, Test2
_C.TEST.INLINE = True
_C.TEST.CROSSLINE = True


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

