#!/bin/bash
export PYTHONPATH=/data/home/mat/repos/DeepSeismic/interpretation:$PYTHONPATH
python test.py --cfg "configs/patch_deconvnet.yaml"