#!/bin/bash
export PYTHONPATH=/workspace:$PYTHONPATH
python -m torch.distributed.launch --nproc_per_node=2 train.py --cfg configs/unet.yaml