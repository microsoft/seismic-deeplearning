#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
nohup python train.py --cfg "configs/seresnet_unet.yaml" > train.log 2>&1
