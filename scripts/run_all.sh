conda activate seismic-interpretation

cd experiments/interpretation/penobscot/local

# Penobscot seresnet unet with section depth
export CUDA_VISIBLE_DEVICES=0
nohup time python train.py \
    'DATASET.ROOT' '/home/maxkaz/data/penobscot' \
    'TRAIN.DEPTH' 'section' \
    'OUTPUT_DIR' 'output' 'TRAIN.MODEL_DIR' 'section_depth' \
    --cfg "configs/seresnet.yaml" > seresnet_unet.log 2>&1 &
# wait for python to pick up the runtime env before switching it
sleep 1

# Penobscot hrnet with section depth
export CUDA_VISIBLE_DEVICES=1
nohup time python train.py \
    'DATASET.ROOT' '/home/maxkaz/data/penobscot' \
    'TRAIN.DEPTH' 'section' \
    'MODEL.PRETRAINED' '/home/maxkaz/models/hrnetv2_w48_imagenet_pretrained.pth' 
    'OUTPUT_DIR' 'output' 'TRAIN.MODEL_DIR' 'section_depth' \
    --cfg=configs/hrnet.yaml --debug > hrnet.log 2>&1 &
# wait for python to pick up the runtime env before switching it
sleep 1

cd ../../dutchf3_patch/local
MODEL_NAME='seresnet_unet'
export CUDA_VISIBLE_DEVICES=2; nohup time python train.py 'DATASET.ROOT'
'/home/maxkaz/data/dutchf3/data' 'MODEL.NAME' "${MODEL_NAME}" 
--cfg
"configs/${MODEL_NAME}.yaml" > ${MODEL_NAME}.log 2>&1 &
MODEL_NAME='hrnet'
export CUDA_VISIBLE_DEVICES=3; nohup time python train.py 'DATASET.ROOT'
'/home/maxkaz/data/dutchf3/data' 'MODEL.NAME' "${MODEL_NAME}"
'MODEL.PRETRAINED' '/home/maxkaz/models/hrnetv2_w48_imagenet_pretrained.pth'
--cfg "configs/${MODEL_NAME}.yaml" > ${MODEL_NAME}.log 2>&1 &
MODEL_NAME='patch_deconvnet_skip'
export CUDA_VISIBLE_DEVICES=4; nohup time python train.py 'DATASET.ROOT'
'/home/maxkaz/data/dutchf3/data' 'MODEL.NAME' "${MODEL_NAME}" --cfg
"configs/${MODEL_NAME}.yaml" > ${MODEL_NAME}.log 2>&1 &
MODEL_NAME='patch_deconvnet'
export CUDA_VISIBLE_DEVICES=5; nohup time python train.py 'DATASET.ROOT'
'/home/maxkaz/data/dutchf3/data' 'MODEL.NAME' "${MODEL_NAME}" --cfg
"configs/${MODEL_NAME}.yaml" > ${MODEL_NAME}.log 2>&1 &
MODEL_NAME='unet'
export CUDA_VISIBLE_DEVICES=6; nohup time python train.py 'DATASET.ROOT'
'/home/maxkaz/data/dutchf3/data' 'MODEL.NAME' "${MODEL_NAME}" --cfg
"configs/${MODEL_NAME}.yaml" > ${MODEL_NAME}.log 2>&1 &

cd ../../dutchf3_section/local
MODEL_NAME='section_deconvnet_skip'
export CUDA_VISIBLE_DEVICES=7; nohup time python train.py 'DATASET.ROOT'
'/home/maxkaz/data/dutchf3/data' 'MODEL.NAME' "${MODEL_NAME}" --cfg
"configs/${MODEL_NAME}.yaml" > ${MODEL_NAME}.log 2>&1 &

unset CUDA_VISIBLE_DEVICES

echo "LAUNCHED ALL LOCAL JOBS"

