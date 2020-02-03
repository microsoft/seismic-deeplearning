#!/bin/bash

# specify absolute locations to your models and data
MODEL_ROOT="/your/model/root"
DATA_ROOT="/your/data/root"

# specify pretrained HRNet backbone
PRETRAINED_HRNET="${MODEL_ROOT}/hrnetv2_w48_imagenet_pretrained.pth"
DATA_F3="${DATA_ROOT}/dutchf3/data"
DATA_PENOBSCOT="${DATA_ROOT}/penobscot"

# subdirectory where results are written
OUTPUT_DIR='output'

# bug to fix conda not launching from a bash shell
source /data/anaconda/etc/profile.d/conda.sh
conda activate seismic-interpretation

cd experiments/interpretation/penobscot/local

# Penobscot seresnet unet with section depth
export CUDA_VISIBLE_DEVICES=0
nohup time python train.py \
    'DATASET.ROOT' "${DATA_PENOBSCOT}" \
    'TRAIN.DEPTH' 'section' \
    'OUTPUT_DIR' "${OUTPUT_DIR}" 'TRAIN.MODEL_DIR' 'section_depth' \
    --cfg "configs/seresnet_unet.yaml" > seresnet_unet.log 2>&1 &
# wait for python to pick up the runtime env before switching it
sleep 1

# Penobscot hrnet with section depth
export CUDA_VISIBLE_DEVICES=1
nohup time python train.py \
    'DATASET.ROOT' "${DATA_PENOBSCOT}" \
    'TRAIN.DEPTH' 'section' \
    'MODEL.PRETRAINED' "${PRETRAINED_HRNET}" \
    'OUTPUT_DIR' "${OUTPUT_DIR}" 'TRAIN.MODEL_DIR' 'section_depth' \
    --cfg=configs/hrnet.yaml > hrnet.log 2>&1 &
# wait for python to pick up the runtime env before switching it
sleep 1

cd ../../dutchf3_patch/local

# patch based without skip connections
export CUDA_VISIBLE_DEVICES=2
nohup time python train.py \
    'DATASET.ROOT' "${DATA_F3}" \
    'TRAIN.DEPTH' 'none' \
    'OUTPUT_DIR' "${OUTPUT_DIR}" 'TRAIN.MODEL_DIR' 'no_depth' \
    --cfg=configs/patch_deconvnet.yaml > patch_deconvnet.log 2>&1 &
# wait for python to pick up the runtime env before switching it
sleep 1

# patch based with skip connections
export CUDA_VISIBLE_DEVICES=3
nohup time python train.py \
    'DATASET.ROOT' "${DATA_F3}" \
    'TRAIN.DEPTH' 'none' \
    'OUTPUT_DIR' "${OUTPUT_DIR}" 'TRAIN.MODEL_DIR' 'no_depth' \
    --cfg=configs/patch_deconvnet_skip.yaml > patch_deconvnet_skip.log 2>&1 &
# wait for python to pick up the runtime env before switching it
sleep 1

# squeeze excitation resnet unet + section depth
export CUDA_VISIBLE_DEVICES=4
nohup time python train.py \
    'DATASET.ROOT' "${DATA_F3}" \
    'TRAIN.DEPTH' 'section' \
    'OUTPUT_DIR' "${OUTPUT_DIR}" 'TRAIN.MODEL_DIR' 'section_depth' \
    --cfg=configs/seresnet_unet.yaml > seresnet_unet.log 2>&1 &
# wait for python to pick up the runtime env before switching it
sleep 1

# HRNet + patch depth
export CUDA_VISIBLE_DEVICES=5
nohup time python train.py \
    'DATASET.ROOT' "${DATA_F3}" \
    'TRAIN.DEPTH' 'patch' \
    'MODEL.PRETRAINED' "${PRETRAINED_HRNET}" \
    'OUTPUT_DIR' "${OUTPUT_DIR}" 'TRAIN.MODEL_DIR' 'patch_depth' \
    --cfg=configs/hrnet.yaml > hrnet_patch.log 2>&1 &
# wait for python to pick up the runtime env before switching it
sleep 1

# HRNet + section depth
export CUDA_VISIBLE_DEVICES=6
nohup time python train.py \
    'DATASET.ROOT' "${DATA_F3}" \
    'TRAIN.DEPTH' 'section' \
    'MODEL.PRETRAINED' "${PRETRAINED_HRNET}" \
    'OUTPUT_DIR' "${OUTPUT_DIR}" 'TRAIN.MODEL_DIR' 'section_depth' \
    --cfg=configs/hrnet.yaml > hrnet_section.log 2>&1 &
# wait for python to pick up the runtime env before switching it
sleep 1

cd ../../dutchf3_section/local

# and finally do a section-based model for comparison 
# (deconv with skip connections and no depth)
export CUDA_VISIBLE_DEVICES=7
nohup time python train.py \
    'DATASET.ROOT' "${DATA_F3}" \
    'TRAIN.DEPTH' 'none' \
    'OUTPUT_DIR' "${OUTPUT_DIR}" 'TRAIN.MODEL_DIR' 'no_depth' \
    --cfg=configs/section_deconvnet_skip.yaml > section_deconvnet_skip.log 2>&1 &
# wait for python to pick up the runtime env before switching it
sleep 1

unset CUDA_VISIBLE_DEVICES

echo "LAUNCHED ALL LOCAL JOBS"

