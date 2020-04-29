#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved. 
# Licensed under the MIT License. 

# Downloads and prepares the data for the rest of the builds to use

# Download the Dutch F3 dataset and extract
if [ -z $1 ]; then 
    echo "You need to specify a download location for the data"
    exit 1;
fi;

DATA_ROOT=$1

source activate seismic-interpretation

# these have to match the rest of the build jobs unless we want to
# define this in ADO pipelines
DATA_CHECKERBOARD="${DATA_ROOT}/checkerboard"
DATA_F3="${DATA_ROOT}/dutch_f3"
DATA_PENOBSCOT="${DATA_ROOT}/penobscot"

# remove data
if [ -d ${DATA_ROOT} ]; then
    echo "Erasing data root dir ${DATA_ROOT}"
    rm -rf "${DATA_ROOT}"
fi
mkdir -p "${DATA_F3}"
mkdir -p "${DATA_PENOBSCOT}"

# test download scripts in parallel
./scripts/download_penobscot.sh "${DATA_PENOBSCOT}" &
./scripts/download_dutch_f3.sh "${DATA_F3}" &
wait

# change imposed by download script
DATA_F3="${DATA_F3}/data"

cd scripts

python gen_checkerboard.py --dataroot ${DATA_F3} --dataout ${DATA_CHECKERBOARD}

# finished data download and generation

# test preprocessing scripts
python prepare_penobscot.py split_inline --data-dir=${DATA_PENOBSCOT} --val-ratio=.1 --test-ratio=.2
python prepare_dutchf3.py split_train_val section --data_dir=${DATA_F3} --label_file=train/train_labels.npy --output_dir=splits --split_direction=both
python prepare_dutchf3.py split_train_val patch   --data_dir=${DATA_F3} --label_file=train/train_labels.npy --output_dir=splits --stride=50 --patch_size=100 --split_direction=both

DATA_CHECKERBOARD="${DATA_CHECKERBOARD}/data"
# repeat for checkerboard dataset
python prepare_dutchf3.py split_train_val section --data_dir=${DATA_CHECKERBOARD} --label_file=train/train_labels.npy --output_dir=splits --split_direction=both
python prepare_dutchf3.py split_train_val patch   --data_dir=${DATA_CHECKERBOARD} --label_file=train/train_labels.npy --output_dir=splits --stride=50 --patch_size=100 --split_direction=both
