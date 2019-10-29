#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Script to kill multiple tmux windows

tmux killw -t hrnet
tmux killw -t hrnet_section_depth
tmux killw -t hrnet_patch_depth

tmux killw -t seresnet_unet
tmux killw -t seresnet_unet_section_depth
tmux killw -t seresnet_unet_patch_depth