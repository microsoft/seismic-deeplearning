#!/bin/bash
export PYTHONPATH=/data/home/mat/repos/ignite_test:$PYTHONPATH
python train.py --cfg "/data/home/mat/repos/ignite_test/experiments/segmentation/tgs_salt/local/configs/unet.yaml"