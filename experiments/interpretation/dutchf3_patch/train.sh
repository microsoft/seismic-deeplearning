#!/bin/bash
nohup python train.py --cfg "configs/seresnet_unet.yaml" > train.log 2>&1
