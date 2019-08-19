#!/bin/bash
export PYTHONPATH=/data/home/mat/repos/ignite_test:$PYTHONPATH
python -m torch.distributed.launch --nproc_per_node=8 train.py --cfg configs/unet.yaml