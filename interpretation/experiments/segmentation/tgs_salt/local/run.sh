#!/bin/bash
export PYTHONPATH=/data/home/mat/repos/DeepSeismic/interpretation:$PYTHONPATH
python train.py --cfg "/data/home/mat/repos/DeepSeismic/interpretation/experiments/segmentation/tgs_salt/local/configs/unet.yaml"