#!/bin/bash

conda activate seismic-interpretation

NGPU=8
export
PYTHONPATH=/home/maxkaz/repos/seismic-deeplearning/interpretation:$PYTHONPATH
cd experiments/interpretation/dutchf3_patch/distributed/

MODEL_NAME='seresnet_unet'
nohup time python -m torch.distributed.launch --nproc_per_node=${NGPU} train.py
'DATASET.ROOT' '/home/maxkaz/data/dutchf3/data' 'MODEL.NAME' "${MODEL_NAME}"
--cfg "configs/${MODEL_NAME}.yaml" > ${MODEL_NAME}.log 2>&1 &
MODEL_NAME='hrnet'
nohup time python -m torch.distributed.launch --nproc_per_node=${NGPU} train.py
'DATASET.ROOT' '/home/maxkaz/data/dutchf3/data' 'MODEL.NAME' "${MODEL_NAME}"
'MODEL.PRETRAINED' '/home/maxkaz/models/hrnetv2_w48_imagenet_pretrained.pth'
--cfg "configs/${MODEL_NAME}.yaml" > ${MODEL_NAME}.log 2>&1 &
MODEL_NAME='patch_deconvnet_skip'
nohup time python -m torch.distributed.launch --nproc_per_node=${NGPU} train.py
'DATASET.ROOT' '/home/maxkaz/data/dutchf3/data' 'MODEL.NAME' "${MODEL_NAME}"
--cfg "configs/${MODEL_NAME}.yaml" > ${MODEL_NAME}.log 2>&1 &
MODEL_NAME='patch_deconvnet'
nohup time python -m torch.distributed.launch --nproc_per_node=${NGPU} train.py
'DATASET.ROOT' '/home/maxkaz/data/dutchf3/data' 'MODEL.NAME' "${MODEL_NAME}"
--cfg "configs/${MODEL_NAME}.yaml" > ${MODEL_NAME}.log 2>&1 &
MODEL_NAME='unet'
nohup time python -m torch.distributed.launch --nproc_per_node=${NGPU} train.py
'DATASET.ROOT' '/home/maxkaz/data/dutchf3/data' 'MODEL.NAME' "${MODEL_NAME}"
--cfg "configs/${MODEL_NAME}.yaml" > ${MODEL_NAME}.log 2>&1 &

echo "TADA"

