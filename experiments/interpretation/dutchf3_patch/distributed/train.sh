#!/bin/bash
export PYTHONPATH=/home/username/seismic-deeplearning/:$PYTHONPATH
python -m torch.distributed.launch --nproc_per_node=4 train.py --cfg configs/hrnet.yaml