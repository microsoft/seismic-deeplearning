#!/bin/bash

# specify absolute locations to your models, data and storage
MODEL_ROOT="/your/model/root"
DATA_ROOT="/your/data/root"
STORAGE_ROOT="/your/storage/root"

# specify pretrained HRNet backbone
PRETRAINED_HRNET="${MODEL_ROOT}/hrnetv2_w48_imagenet_pretrained.pth"
DATA_F3="${DATA_ROOT}/dutchf3/data"
DATA_PENOBSCOT="${DATA_ROOT}/penobscot"
# name of your git branch which you ran the training code from
BRANCH="your/git/branch/with/slashes/if/they/exist/in/branch/name"

# name of directory where results are kept
OUTPUT_DIR="output"

# directory where to copy pre-trained models to
OUTPUT_PRETRAINED="${STORAGE_ROOT}/pretrained_models/"

if [ -d ${OUTPUT_PRETRAINED} ]; then
    echo "erasing pre-trained models in ${OUTPUT_PRETRAINED}"
    rm -rf "${OUTPUT_PRETRAINED}"
fi

mkdir -p "${OUTPUT_PRETRAINED}"
echo "Pre-trained models will be copied to ${OUTPUT_PRETRAINED}"

# bug to fix conda not launching from a bash shell
source /data/anaconda/etc/profile.d/conda.sh
conda activate seismic-interpretation

cd experiments/interpretation/penobscot/local

# Penobscot seresnet unet with section depth
export CUDA_VISIBLE_DEVICES=0
CONFIG_NAME='seresnet_unet'
# master
# model=$(ls -td ${OUTPUT_DIR}/${BRANCH}/*/resnet_unet/*/section_depth/*.pth | head -1)
# new staging structure
model=$(ls -td ${OUTPUT_DIR}/${BRANCH}/*/${CONFIG_NAME}/section_depth/*/*.pth | head -1)
cp $model ${OUTPUT_PRETRAINED}/penobscot_seresnetunet_patch_section_depth.pth
nohup time python test.py \
    'DATASET.ROOT' "${DATA_PENOBSCOT}" 'TEST.MODEL_PATH' "${model}" \
    --cfg "configs/${CONFIG_NAME}.yaml" > ${CONFIG_NAME}_test.log 2>&1 &
sleep 1

# Penobscot hrnet with section depth
export CUDA_VISIBLE_DEVICES=1
CONFIG_NAME='hrnet'
# master
# model=$(ls -td ${OUTPUT_DIR}/${BRANCH}/*/seg_hrnet/*/section_depth/*.pth | head -1)
# new staging structure
model=$(ls -td ${OUTPUT_DIR}/${BRANCH}/*/${CONFIG_NAME}/section_depth/*/*.pth | head -1)
cp $model ${OUTPUT_PRETRAINED}/penobscot_hrnet_patch_section_depth.pth
nohup time python test.py \
    'DATASET.ROOT' "${DATA_PENOBSCOT}" 'TEST.MODEL_PATH' "${model}" \
    'MODEL.PRETRAINED' "${PRETRAINED_HRNET}" \
    --cfg "configs/${CONFIG_NAME}.yaml" > ${CONFIG_NAME}_test.log 2>&1 &
sleep 1

cd ../../dutchf3_patch/local

# patch based without skip connections
export CUDA_VISIBLE_DEVICES=2
CONFIG_NAME='patch_deconvnet'
model=$(ls -td ${OUTPUT_DIR}/${BRANCH}/*/${CONFIG_NAME}/no_depth/*/*.pth | head -1)
cp $model ${OUTPUT_PRETRAINED}/dutchf3_deconvnet_patch_no_depth.pth
nohup time python test.py \
    'DATASET.ROOT' "${DATA_F3}" 'TEST.MODEL_PATH' "${model}" \
    --cfg "configs/${CONFIG_NAME}.yaml" > ${CONFIG_NAME}_test.log 2>&1 &
sleep 1

# patch based with skip connections
export CUDA_VISIBLE_DEVICES=3
CONFIG_NAME='patch_deconvnet_skip'
model=$(ls -td ${OUTPUT_DIR}/${BRANCH}/*/${CONFIG_NAME}/no_depth/*/*.pth | head -1)
cp $model ${OUTPUT_PRETRAINED}/dutchf3_deconvnetskip_patch_no_depth.pth
nohup time python test.py \
    'DATASET.ROOT' "${DATA_F3}" 'TEST.MODEL_PATH' "${model}" \
    --cfg "configs/${CONFIG_NAME}.yaml" > ${CONFIG_NAME}_test.log 2>&1 &
sleep 1

# squeeze excitation resnet unet + section depth
export CUDA_VISIBLE_DEVICES=4
CONFIG_NAME='seresnet_unet'
# master
# model=$(ls -td ${OUTPUT_DIR}/${BRANCH}/*/resnet_unet/*/section_depth/*.pth | head -1)
# staging
model=$(ls -td ${OUTPUT_DIR}/${BRANCH}/*/${CONFIG_NAME}/section_depth/*/*.pth | head -1)
cp $model ${OUTPUT_PRETRAINED}/dutchf3_seresnetunet_patch_section_depth.pth
nohup time python test.py \
    'DATASET.ROOT' "${DATA_F3}" 'TEST.MODEL_PATH' "${model}" \
    --cfg "configs/${CONFIG_NAME}.yaml" > ${CONFIG_NAME}_test.log 2>&1 &
sleep 1

# HRNet + patch depth
export CUDA_VISIBLE_DEVICES=5
CONFIG_NAME='hrnet'
# master
# model=$(ls -td ${OUTPUT_DIR}/${BRANCH}/*/seg_hrnet/*/patch_depth/*.pth | head -1)
# staging
model=$(ls -td ${OUTPUT_DIR}/${BRANCH}/*/$CONFIG_NAME/patch_depth/*/*.pth | head -1)
cp $model ${OUTPUT_PRETRAINED}/dutchf3_hrnet_patch_patch_depth.pth
nohup time python test.py \
    'DATASET.ROOT' "${DATA_F3}" 'TEST.MODEL_PATH' "${model}" \
    'MODEL.PRETRAINED' "${PRETRAINED_HRNET}" \
    --cfg "configs/${CONFIG_NAME}.yaml" > ${CONFIG_NAME}_patch_test.log 2>&1 &
sleep 1

# HRNet + section depth
export CUDA_VISIBLE_DEVICES=6
CONFIG_NAME='hrnet'
# master
# model=$(ls -td ${OUTPUT_DIR}/${BRANCH}/*/seg_hrnet/*/section_depth/*.pth | head -1)
# staging
model=$(ls -td ${OUTPUT_DIR}/${BRANCH}/*/${CONFIG_NAME}/section_depth/*/*.pth | head -1)
cp $model ${OUTPUT_PRETRAINED}/dutchf3_hrnet_patch_section_depth.pth
nohup time python test.py \
    'DATASET.ROOT' "${DATA_F3}" 'TEST.MODEL_PATH' "${model}" \
    'MODEL.PRETRAINED' "${PRETRAINED_HRNET}" \
    --cfg "configs/${CONFIG_NAME}.yaml" > ${CONFIG_NAME}_section_test.log 2>&1 &
sleep 1

cd ../../dutchf3_section/local

# and finally do a section-based model for comparison 
# (deconv with skip connections and no depth)
export CUDA_VISIBLE_DEVICES=7
CONFIG_NAME='section_deconvnet_skip'
model=$(ls -td ${OUTPUT_DIR}/${BRANCH}/*/${CONFIG_NAME}/no_depth/*/*.pth | head -1)
cp $model ${OUTPUT_PRETRAINED}/dutchf3_deconvnetskip_section_no_depth.pth
nohup time python test.py \
    'DATASET.ROOT' "${DATA_F3}" 'TEST.MODEL_PATH' "${model}" \
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
model=$(ls -td ../distributed/${OUTPUT_DIR}/${BRANCH}/*/${CONFIG_NAME}/*/no_depth/*.pth | head -1)
cp $model ${OUTPUT_PRETRAINED}/dutchf3_deconvnet_patch_no_depth_distributed.pth
nohup time python test.py \
    'DATASET.ROOT' "${DATA_F3}" 'TEST.MODEL_PATH' "${model}" \
    --cfg "configs/${CONFIG_NAME}.yaml" > ${CONFIG_NAME}_distributed_test.log 2>&1 &
sleep 1

# patch based with skip connections
export CUDA_VISIBLE_DEVICES=3
CONFIG_NAME='patch_deconvnet_skip'
model=$(ls -td ../distributed/${OUTPUT_DIR}/${BRANCH}/*/${CONFIG_NAME}/*/no_depth/*.pth | head -1)
cp $model ${OUTPUT_PRETRAINED}/dutchf3_deconvnetskip_patch_no_depth_distributed.pth
nohup time python test.py \
    'DATASET.ROOT' "${DATA_F3}" 'TEST.MODEL_PATH' "${model}" \
    --cfg "configs/${CONFIG_NAME}.yaml" > ${CONFIG_NAME}_distributed_test.log 2>&1 &
sleep 1

# squeeze excitation resnet unet + section depth
export CUDA_VISIBLE_DEVICES=4
CONFIG_NAME='seresnet_unet'
model=$(ls -td ../distributed/${OUTPUT_DIR}/${BRANCH}/*/resnet_unet/*/section_depth/*.pth | head -1)
cp $model ${OUTPUT_PRETRAINED}/dutchf3_seresnetunet_patch_section_depth_distributed.pth
nohup time python test.py \
    'DATASET.ROOT' "${DATA_F3}" 'TEST.MODEL_PATH' "${model}" \
    --cfg "configs/${CONFIG_NAME}.yaml" > ${CONFIG_NAME}_distributed_test.log 2>&1 &
sleep 1

# HRNet + patch depth
export CUDA_VISIBLE_DEVICES=5
CONFIG_NAME='hrnet'
model=$(ls -td ../distributed/${OUTPUT_DIR}/${BRANCH}/*/seg_hrnet/*/patch_depth/*.pth | head -1)
cp $model ${OUTPUT_PRETRAINED}/dutchf3_hrnet_patch_patch_depth_distributed.pth
nohup time python test.py \
    'DATASET.ROOT' "${DATA_F3}" 'TEST.MODEL_PATH' "${model}" \
    'MODEL.PRETRAINED' "${PRETRAINED_HRNET}" \
    --cfg "configs/${CONFIG_NAME}.yaml" > ${CONFIG_NAME}_distributed_test.log 2>&1 &
sleep 1

# HRNet + section depth
export CUDA_VISIBLE_DEVICES=6
CONFIG_NAME='hrnet'
model=$(ls -td ../distributed/${OUTPUT_DIR}/${BRANCH}/*/seg_hrnet/*/section_depth/*.pth | head -1)
cp $model ${OUTPUT_PRETRAINED}/dutchf3_hrnet_patch_section_depth_distributed.pth
nohup time python test.py \
    'DATASET.ROOT' "${DATA_F3}" 'TEST.MODEL_PATH' "${model}" \
    'MODEL.PRETRAINED' "${PRETRAINED_HRNET}" \
    --cfg "configs/${CONFIG_NAME}.yaml" > ${CONFIG_NAME}_distributed_test.log 2>&1 &
sleep 1

echo "Waiting for all distributed runs to finish"

wait

echo "TADA"
