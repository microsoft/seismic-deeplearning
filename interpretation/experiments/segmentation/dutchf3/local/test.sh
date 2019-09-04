#!/bin/bash
export PYTHONPATH=/data/home/mat/repos/DeepSeismic/interpretation:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
python test.py --cfg "configs/seresnet_unet.yaml"