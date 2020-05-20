#!/bin/bash

source activate seismic-interpretation

# Patch_Size 100: Patch vs Section Depth
python scripts/prepare_dutchf3.py split_train_val patch --data_dir=/mnt/dutch --stride=50 --patch_size=100 --split_direction=both
python train.py OUTPUT_DIR /data/output/hrnet_patch TRAIN.DEPTH patch TRAIN.PATCH_SIZE 100 --cfg 'configs/hrnet.yaml'
python train.py OUTPUT_DIR /data/output/hrnet_section TRAIN.DEPTH section TRAIN.PATCH_SIZE 100 --cfg 'configs/hrnet.yaml'

# Patch_Size 150: Patch vs Section Depth
python scripts/prepare_dutchf3.py split_train_val patch --data_dir=/mnt/dutch --stride=50 --patch_size=150 --split_direction=both
python train.py OUTPUT_DIR /data/output/hrnet_patch TRAIN.DEPTH patch TRAIN.PATCH_SIZE 150 --cfg 'configs/hrnet.yaml'
python train.py OUTPUT_DIR /data/output/hrnet_section TRAIN.DEPTH section TRAIN.PATCH_SIZE 150 --cfg 'configs/hrnet.yaml'

# Patch_Size 200: Patch vs Section Depth
python scripts/prepare_dutchf3.py split_train_val patch --data_dir=/mnt/dutch --stride=50 --patch_size=200 --split_direction=both
python train.py OUTPUT_DIR /data/output/hrnet_patch TRAIN.DEPTH patch TRAIN.PATCH_SIZE 200 --cfg 'configs/hrnet.yaml'
python train.py OUTPUT_DIR /data/output/hrnet_section TRAIN.DEPTH section TRAIN.PATCH_SIZE 200 --cfg 'configs/hrnet.yaml'

# Patch_Size 250: Patch vs Section Depth
python scripts/prepare_dutchf3.py split_train_val patch --data_dir=/mnt/dutch --stride=50 --patch_size=250 --split_direction=both
python train.py OUTPUT_DIR /data/output/hrnet_patch TRAIN.DEPTH patch TRAIN.PATCH_SIZE 250 TRAIN.AUGMENTATIONS.RESIZE.HEIGHT 250 TRAIN.AUGMENTATIONS.RESIZE.WIDTH 250 --cfg 'configs/hrnet.yaml'
python train.py OUTPUT_DIR /data/output/hrnet_section TRAIN.DEPTH section TRAIN.PATCH_SIZE 250 TRAIN.AUGMENTATIONS.RESIZE.HEIGHT 250 TRAIN.AUGMENTATIONS.RESIZE.WIDTH 250 --cfg 'configs/hrnet.yaml'

