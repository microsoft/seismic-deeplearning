#!/bin/bash
export PYTHONPATH=/data/home/mat/repos/DeepSeismic/interpretation:$PYTHONPATH
python train.py --cfg "/data/home/mat/repos/DeepSeismic/interpretation/experiments/segmentation/dutchf3/local/configs/unet.yaml"