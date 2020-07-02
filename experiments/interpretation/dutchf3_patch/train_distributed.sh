#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
NGPUS=$(nvidia-smi -L | wc -l)
if [ "$NGPUS" -lt "2" ]; then
    echo "ERROR: cannot run distributed training without 2 or more GPUs."
    exit 1
fi
nohup python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py \
     --distributed --cfg "configs/seresnet_unet.yaml" > train_distributed.log 2>&1 &
