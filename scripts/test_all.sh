#!/bin/bash

# DATA_F3='/home/alfred/data/dutch_f3/data'
# DATA_PENOBSCOT='/home/maxkaz/data/penobscot'
DATA_F3='/storage/data/dutchf3/data'
DATA_PENOBSCOT='/storage/data/penobscot'
# name of your git branch which you ran the training code from
BRANCH="maxkaz/names"
# name of directory where results are kept
OUTPUT_DIR="output"

# bug to fix conda not launching from a bash shell
source /data/anaconda/etc/profile.d/conda.sh
conda activate seismic-interpretation

cd experiments/interpretation/penobscot/local

# Penobscot seresnet unet with section depth
export CUDA_VISIBLE_DEVICES=0
CONFIG_NAME='seresnet_unet'
model=$(ls -td ${OUTPUT_DIR}/${BRANCH}/*/${CONFIG_NAME}/section_depth/*/*.pth | head -1)
nohup time python test.py \
    'DATASET.ROOT' "${DATASET_ROOT}" 'TEST.MODEL_PATH' "${model}" \
    --cfg "configs/${CONFIG_NAME}.yaml" > ${CONFIG_NAME}_test.log 2>&1 &
sleep 1

# Penobscot hrnet with section depth
export CUDA_VISIBLE_DEVICES=1
CONFIG_NAME='hrnet'
model=$(ls -td ${OUTPUT_DIR}/${BRANCH}/*/${CONFIG_NAME}/section_depth/*/*.pth | head -1)
nohup time python test.py \
    'DATASET.ROOT' "${DATASET_ROOT}" 'TEST.MODEL_PATH' "${model}" \
    --cfg "configs/${CONFIG_NAME}.yaml" > ${CONFIG_NAME}_test.log 2>&1 &
sleep 1

cd ../../dutchf3_patch/local

# patch based without skip connections
export CUDA_VISIBLE_DEVICES=2
CONFIG_NAME='patch_deconvnet'
model=$(ls -td ${OUTPUT_DIR}/${BRANCH}/*/${CONFIG_NAME}/no_depth/*/*.pth | head -1)
nohup time python test.py \
    'DATASET.ROOT' "${DATASET_ROOT}" 'TEST.MODEL_PATH' "${model}" \
    --cfg "configs/${CONFIG_NAME}.yaml" > ${CONFIG_NAME}_test.log 2>&1 &
sleep 1

# patch based with skip connections
export CUDA_VISIBLE_DEVICES=3
CONFIG_NAME='patch_deconvnet_skip'
model=$(ls -td ${OUTPUT_DIR}/${BRANCH}/*/${CONFIG_NAME}/no_depth/*/*.pth | head -1)
nohup time python test.py \
    'DATASET.ROOT' "${DATASET_ROOT}" 'TEST.MODEL_PATH' "${model}" \
    --cfg "configs/${CONFIG_NAME}.yaml" > ${CONFIG_NAME}_test.log 2>&1 &
sleep 1

# squeeze excitation resnet unet + section depth
export CUDA_VISIBLE_DEVICES=4
CONFIG_NAME='seresnet_unet'
model=$(ls -td ${OUTPUT_DIR}/${BRANCH}/*/${CONFIG_NAME}/section_depth/*/*.pth | head -1)
nohup time python test.py \
    'DATASET.ROOT' "${DATASET_ROOT}" 'TEST.MODEL_PATH' "${model}" \
    --cfg "configs/${CONFIG_NAME}.yaml" > ${CONFIG_NAME}_test.log 2>&1 &
sleep 1

# HRNet + patch depth
export CUDA_VISIBLE_DEVICES=5
CONFIG_NAME='hrnet'
model=$(ls -td ${OUTPUT_DIR}/${BRANCH}/*/${CONFIG_NAME}/patch_depth/*/*.pth | head -1)
nohup time python test.py \
    'DATASET.ROOT' "${DATASET_ROOT}" 'TEST.MODEL_PATH' "${model}" \
    --cfg "configs/${CONFIG_NAME}.yaml" > ${CONFIG_NAME}_test.log 2>&1 &
sleep 1

# HRNet + section depth
export CUDA_VISIBLE_DEVICES=6
CONFIG_NAME='hrnet'
model=$(ls -td ${OUTPUT_DIR}/${BRANCH}/*/${CONFIG_NAME}/section_depth/*/*.pth | head -1)
nohup time python test.py \
    'DATASET.ROOT' "${DATASET_ROOT}" 'TEST.MODEL_PATH' "${model}" \
    --cfg "configs/${CONFIG_NAME}.yaml" > ${CONFIG_NAME}_test.log 2>&1 &
sleep 1

cd ../../dutchf3_section/local

# and finally do a section-based model for comparison 
# (deconv with skip connections and no depth)
export CUDA_VISIBLE_DEVICES=7
CONFIG_NAME='section_deconvnet_skip'
model=$(ls -td ${OUTPUT_DIR}/${BRANCH}/*/${CONFIG_NAME}/no_depth/*/*.pth | head -1)
nohup time python test.py \
    'DATASET.ROOT' "${DATASET_ROOT}" 'TEST.MODEL_PATH' "${model}" \
    --cfg "configs/${CONFIG_NAME}.yaml" > ${CONFIG_NAME}_test.log 2>&1 &
sleep 1

echo "Waiting for all local runs to finish"
wait

# scoring scripts are in the local folder
# models are in the distributed folder
cd ../../dutchf3_patch/local

# patch based without skip connections
export CUDA_VISIBLE_DEVICES=2
CONFIG_NAME='patch_deconvnet'
model=$(ls -td ../distributed/${OUTPUT_DIR}/${BRANCH}/*/${CONFIG_NAME}/no_depth/*/*.pth | head -1)
nohup time python test.py \
    'DATASET.ROOT' "${DATASET_ROOT}" 'TEST.MODEL_PATH' "${model}" \
    --cfg "configs/${CONFIG_NAME}.yaml" > ${CONFIG_NAME}_distributed_test.log 2>&1 &
sleep 1

# patch based with skip connections
export CUDA_VISIBLE_DEVICES=3
CONFIG_NAME='patch_deconvnet_skip'
model=$(ls -td ../distributed/${OUTPUT_DIR}/${BRANCH}/*/${CONFIG_NAME}/no_depth/*/*.pth | head -1)
nohup time python test.py \
    'DATASET.ROOT' "${DATASET_ROOT}" 'TEST.MODEL_PATH' "${model}" \
    --cfg "configs/${CONFIG_NAME}.yaml" > ${CONFIG_NAME}_distributed_test.log 2>&1 &
sleep 1

# squeeze excitation resnet unet + section depth
export CUDA_VISIBLE_DEVICES=4
CONFIG_NAME='seresnet_unet'
model=$(ls -td ../distributed/${OUTPUT_DIR}/${BRANCH}/*/${CONFIG_NAME}/section_depth/*/*.pth | head -1)
nohup time python test.py \
    'DATASET.ROOT' "${DATASET_ROOT}" 'TEST.MODEL_PATH' "${model}" \
    --cfg "configs/${CONFIG_NAME}.yaml" > ${CONFIG_NAME}_distributed_test.log 2>&1 &
sleep 1

# HRNet + patch depth
export CUDA_VISIBLE_DEVICES=5
CONFIG_NAME='hrnet'
model=$(ls -td ../distributed/${OUTPUT_DIR}/${BRANCH}/*/${CONFIG_NAME}/patch_depth/*/*.pth | head -1)
nohup time python test.py \
    'DATASET.ROOT' "${DATASET_ROOT}" 'TEST.MODEL_PATH' "${model}" \
    --cfg "configs/${CONFIG_NAME}.yaml" > ${CONFIG_NAME}_distributed_test.log 2>&1 &
sleep 1

# HRNet + section depth
export CUDA_VISIBLE_DEVICES=6
CONFIG_NAME='hrnet'
model=$(ls -td ../distributed/${OUTPUT_DIR}/${BRANCH}/*/${CONFIG_NAME}/section_depth/*/*.pth | head -1)
nohup time python test.py \
    'DATASET.ROOT' "${DATASET_ROOT}" 'TEST.MODEL_PATH' "${model}" \
    --cfg "configs/${CONFIG_NAME}.yaml" > ${CONFIG_NAME}_distributed_test.log 2>&1 &
sleep 1

echo "Waiting for all distributed runs to finish"

wait

echo "TADA"
