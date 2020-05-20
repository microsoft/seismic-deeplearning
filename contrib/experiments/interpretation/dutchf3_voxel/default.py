# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

_C = CN()

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

_C.GPUS = (0,)
_C.OUTPUT_DIR = "output"  # This will be the base directory for all output, such as logs and saved models
_C.LOG_DIR = ""  # This will be a subdirectory inside OUTPUT_DIR
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.LOG_CONFIG = "logging.conf"
_C.SEED = 42
_C.OPENCV_BORDER_CONSTANT = 0

# size of voxel cube: WINDOW_SIZE x WINDOW_SIZE x WINDOW_SIZE; used for 3D models only
_C.WINDOW_SIZE = 65

# DATASET related params
_C.DATASET = CN()
_C.DATASET.NUM_CLASSES = 2
_C.DATASET.ROOT = ""
_C.DATASET.FILENAME = "data.segy"

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = "texture_net"
_C.MODEL.IN_CHANNELS = 1
_C.MODEL.NUM_FILTERS = 50
_C.MODEL.EXTRA = CN(new_allowed=True)

# training
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE_PER_GPU = 32
# number of batches per epoch
_C.TRAIN.BATCH_PER_EPOCH = 10
# total number of epochs
_C.TRAIN.END_EPOCH = 200
_C.TRAIN.LR = 0.01
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 0.0001
_C.TRAIN.DEPTH = "voxel"  # Options are none, patch and section
_C.TRAIN.MODEL_DIR = "models"  # This will be a subdirectory inside OUTPUT_DIR

# validation
_C.VALIDATION = CN()
_C.VALIDATION.BATCH_SIZE_PER_GPU = 32

# TEST
_C.TEST = CN()
_C.TEST.MODEL_PATH = ""
_C.TEST.SPLIT = "Both"  # Can be Both, Test1, Test2


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
