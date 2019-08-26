#!/bin/bash
export PYTHONPATH=/data/home/mat/repos/DeepSeismic/interpretation:$PYTHONPATH
python train.py --cfg "configs/patch_deconvnet.yaml"