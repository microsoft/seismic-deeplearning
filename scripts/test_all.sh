#!/bin/bash

conda activate seismic-interpretation

cd experiments/interpretation/penobscot/local
MODEL_NAME='seresnet_unet'
model=$(ls -td output/*/*/${MODEL_NAME}/*/models/*.pth | head -1)
export CUDA_VISIBLE_DEVICES=0; nohup time python test.py 'DATASET.ROOT'
'/home/maxkaz/data/penobscot' 'TEST.MODEL_PATH' "${model}" --cfg
"configs/${MODEL_NAME}.yaml" > ${MODEL_NAME}_test.log 2>&1 &
MODEL_NAME='hrnet'
model=$(ls -td output/*/*/${MODEL_NAME}/*/models/*.pth | head -1)
export CUDA_VISIBLE_DEVICES=1; nohup time python test.py 'DATASET.ROOT'
'/home/maxkaz/data/penobscot' 'TEST.MODEL_PATH' "${model}" --cfg
"configs/${MODEL_NAME}.yaml" > ${MODEL_NAME}_test.log 2>&1 &

cd ../../dutchf3_patch/local
MODEL_NAME='seresnet_unet'
model=$(ls -td output/*/*/${MODEL_NAME}/*/models/*.pth | head -1)
export CUDA_VISIBLE_DEVICES=2; nohup time python test.py 'DATASET.ROOT'
'/home/maxkaz/data/dutchf3/data' 'TEST.MODEL_PATH' "${model}" --cfg
"configs/${MODEL_NAME}.yaml" > ${MODEL_NAME}_test.log 2>&1 &
MODEL_NAME='hrnet'
model=$(ls -td output/*/*/${MODEL_NAME}/*/models/*.pth | head -1)
export CUDA_VISIBLE_DEVICES=3; nohup time python test.py 'DATASET.ROOT'
'/home/maxkaz/data/dutchf3/data' 'TEST.MODEL_PATH' "${model}" --cfg
"configs/${MODEL_NAME}.yaml" > ${MODEL_NAME}_test.log 2>&1 &
MODEL_NAME='patch_deconvnet_skip'
model=$(ls -td output/*/*/${MODEL_NAME}/*/models/*.pth | head -1)
export CUDA_VISIBLE_DEVICES=4; nohup time python test.py 'DATASET.ROOT'
'/home/maxkaz/data/dutchf3/data' 'TEST.MODEL_PATH' "${model}" --cfg
"configs/${MODEL_NAME}.yaml" > ${MODEL_NAME}_test.log 2>&1 &
MODEL_NAME='patch_deconvnet'
model=$(ls -td output/*/*/${MODEL_NAME}/*/models/*.pth | head -1)
export CUDA_VISIBLE_DEVICES=5; nohup time python test.py 'DATASET.ROOT'
'/home/maxkaz/data/dutchf3/data' 'TEST.MODEL_PATH' "${model}" --cfg
"configs/${MODEL_NAME}.yaml" > ${MODEL_NAME}_test.log 2>&1 &
MODEL_NAME='unet'
model=$(ls -td output/*/*/${MODEL_NAME}/*/models/*.pth | head -1)
export CUDA_VISIBLE_DEVICES=6; nohup time python test.py 'DATASET.ROOT'
'/home/maxkaz/data/dutchf3/data' 'TEST.MODEL_PATH' "${model}" --cfg
"configs/${MODEL_NAME}.yaml" > ${MODEL_NAME}_test.log 2>&1 &

cd ../../dutchf3_section/local
MODEL_NAME='section_deconvnet_skip'
model=$(ls -td output/*/*/${MODEL_NAME}/*/models/*.pth | head -1)
export CUDA_VISIBLE_DEVICES=7; nohup time python test.py 'DATASET.ROOT'
'/home/maxkaz/data/dutchf3/data' 'TEST.MODEL_PATH' "${model}" --cfg
"configs/${MODEL_NAME}.yaml" > ${MODEL_NAME}_test.log 2>&1 &

echo "Waiting for all local runs to finish"

wait

cd ../../dutchf3_patch/local
MODEL_NAME='seresnet_unet'
model=$(ls -td
../../dutchf3_patch/distributed/output/*/*/${MODEL_NAME}/*/models/*.pth | head
-1)
export CUDA_VISIBLE_DEVICES=2; nohup time python test.py 'DATASET.ROOT'
'/home/maxkaz/data/dutchf3/data' 'TEST.MODEL_PATH' "${model}" --cfg
"configs/${MODEL_NAME}.yaml" > ${MODEL_NAME}_test_dist.log 2>&1 &
MODEL_NAME='hrnet'
model=$(ls -td
../../dutchf3_patch/distributed/output/*/*/${MODEL_NAME}/*/models/*.pth | head
-1)
export CUDA_VISIBLE_DEVICES=3; nohup time python test.py 'DATASET.ROOT'
'/home/maxkaz/data/dutchf3/data' 'TEST.MODEL_PATH' "${model}" --cfg
"configs/${MODEL_NAME}.yaml" > ${MODEL_NAME}_test_dist.log 2>&1 &
MODEL_NAME='patch_deconvnet_skip'
model=$(ls -td
../../dutchf3_patch/distributed/output/*/*/${MODEL_NAME}/*/models/*.pth | head
-1)
export CUDA_VISIBLE_DEVICES=4; nohup time python test.py 'DATASET.ROOT'
'/home/maxkaz/data/dutchf3/data' 'TEST.MODEL_PATH' "${model}" --cfg
"configs/${MODEL_NAME}.yaml" > ${MODEL_NAME}_test_dist.log 2>&1 &
MODEL_NAME='patch_deconvnet'
model=$(ls -td
../../dutchf3_patch/distributed/output/*/*/${MODEL_NAME}/*/models/*.pth | head
-1)
export CUDA_VISIBLE_DEVICES=5; nohup time python test.py 'DATASET.ROOT'
'/home/maxkaz/data/dutchf3/data' 'TEST.MODEL_PATH' "${model}" --cfg
"configs/${MODEL_NAME}.yaml" > ${MODEL_NAME}_test_dist.log 2>&1 &
MODEL_NAME='unet'
model=$(ls -td
../../dutchf3_patch/distributed/output/*/*/${MODEL_NAME}/*/models/*.pth | head
-1)
export CUDA_VISIBLE_DEVICES=6; nohup time python test.py 'DATASET.ROOT'
'/home/maxkaz/data/dutchf3/data' 'TEST.MODEL_PATH' "${model}" --cfg
"configs/${MODEL_NAME}.yaml" > ${MODEL_NAME}_test_dist.log 2>&1 &

echo "Waiting for all distributed runs to finish"

wait

unset CUDA_VISIBLE_DEVICES

echo "TADA"

