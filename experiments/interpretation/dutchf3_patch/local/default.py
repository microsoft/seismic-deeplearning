# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

_C = CN()

_C.OUTPUT_DIR = "output"  # This will be the base directory for all output, such as logs and saved models
_C.LOG_DIR = ""  # This will be a subdirectory inside OUTPUT_DIR
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.LOG_CONFIG = "logging.conf"
_C.SEED = 42
_C.OPENCV_BORDER_CONSTANT = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ""
_C.DATASET.NUM_CLASSES = 6
_C.DATASET.CLASS_WEIGHTS = [0.7151, 0.8811, 0.5156, 0.9346, 0.9683, 0.9852]

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = "patch_deconvnet"
_C.MODEL.IN_CHANNELS = 1
_C.MODEL.PRETRAINED = ""
_C.MODEL.EXTRA = CN(new_allowed=True)


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
_C.TRAIN.MODEL_DIR = "models"  # This will be a subdirectory inside OUTPUT_DIR
_C.TRAIN.AUGMENTATION = True
_C.TRAIN.STRIDE = 50
_C.TRAIN.PATCH_SIZE = 99
_C.TRAIN.MEAN = 0.0009997  # 0.0009996710808862074
_C.TRAIN.STD = 0.20977  # 0.20976548783479299  # TODO: Should we apply std scaling?
_C.TRAIN.DEPTH = "none"  # Options are: none, patch, and section

# None adds no depth information and the num of channels remains at 1
# Patch adds depth per patch so is simply the height of that patch from 0 to 1, channels=3
# Section adds depth per section so contains depth information for the whole section, channels=3
_C.TRAIN.AUGMENTATIONS = CN()
_C.TRAIN.AUGMENTATIONS.RESIZE = CN()
_C.TRAIN.AUGMENTATIONS.RESIZE.HEIGHT = 200
_C.TRAIN.AUGMENTATIONS.RESIZE.WIDTH = 200
_C.TRAIN.AUGMENTATIONS.PAD = CN()
_C.TRAIN.AUGMENTATIONS.PAD.HEIGHT = 256
_C.TRAIN.AUGMENTATIONS.PAD.WIDTH = 256

# validation
_C.VALIDATION = CN()
_C.VALIDATION.BATCH_SIZE_PER_GPU = 32

# TEST
_C.TEST = CN()
_C.TEST.MODEL_PATH = ""
_C.TEST.TEST_STRIDE = 10
_C.TEST.SPLIT = "Both"  # Can be Both, Test1, Test2
_C.TEST.INLINE = True
_C.TEST.CROSSLINE = True
_C.TEST.POST_PROCESSING = CN()  # Model output postprocessing
_C.TEST.POST_PROCESSING.SIZE = 128  # Size to interpolate to in pixels
_C.TEST.POST_PROCESSING.CROP_PIXELS = 14  # Number of pixels to crop top, bottom, left and right


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
