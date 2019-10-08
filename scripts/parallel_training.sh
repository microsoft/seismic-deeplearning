#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Script to run multiple models in parallel on multi-gpu machine

workspace=../experiments/segmentation/penobscot/local
tmux neww -d -n hrnet
tmux neww -d -n hrnet_section_depth
tmux neww -d -n hrnet_patch_depth
tmux neww -d -n seresnet_unet
tmux neww -d -n seresnet_unet_section_depth
tmux neww -d -n seresnet_unet_patch_depth

tmux send -t hrnet "source activate seismic-interpretation && cd ${workspace}" ENTER
tmux send -t hrnet "CUDA_VISIBLE_DEVICES=0 python train.py OUTPUT_DIR /data/output/hrnet --cfg 'configs/hrnet.yaml'" ENTER

tmux send -t hrnet_patch_depth "source activate seismic-interpretation && cd ${workspace}" ENTER
tmux send -t hrnet_patch_depth "CUDA_VISIBLE_DEVICES=1 python train.py OUTPUT_DIR /data/output/hrnet_patch TRAIN.DEPTH patch --cfg 'configs/hrnet.yaml'" ENTER

tmux send -t hrnet_section_depth "source activate seismic-interpretation && cd ${workspace}" ENTER
tmux send -t hrnet_section_depth "CUDA_VISIBLE_DEVICES=2 python train.py OUTPUT_DIR /data/output/hrnet_section TRAIN.DEPTH section --cfg 'configs/hrnet.yaml'" ENTER

tmux send -t seresnet_unet "source activate seismic-interpretation && cd ${workspace}" ENTER
tmux send -t seresnet_unet "CUDA_VISIBLE_DEVICES=3 python train.py OUTPUT_DIR /data/output/seresnet --cfg 'configs/seresnet_unet.yaml'" ENTER

tmux send -t seresnet_unet_patch_depth "source activate seismic-interpretation && cd ${workspace}" ENTER
tmux send -t seresnet_unet_patch_depth "CUDA_VISIBLE_DEVICES=4 python train.py OUTPUT_DIR /data/output/seresnet_patch TRAIN.DEPTH patch --cfg 'configs/seresnet_unet.yaml'" ENTER

tmux send -t seresnet_unet_section_depth "source activate seismic-interpretation && cd ${workspace}" ENTER
tmux send -t seresnet_unet_section_depth "CUDA_VISIBLE_DEVICES=5 python train.py OUTPUT_DIR /data/output/seresnet_section TRAIN.DEPTH section --cfg 'configs/seresnet_unet.yaml'" ENTER