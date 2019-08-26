#!/bin/bash
export PYTHONPATH=/data/home/mat/repos/DeepSeismic/interpretation:$PYTHONPATH
python -m torch.distributed.launch --nproc_per_node=8 train.py --cfg configs/seresnet_unet.yaml