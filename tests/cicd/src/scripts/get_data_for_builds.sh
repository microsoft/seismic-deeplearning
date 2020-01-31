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
DATA_F3="${DATA_ROOT}/dutch_f3"
DATA_PENOBSCOT="${DATA_ROOT}/penobscot"

# remove data
if [ -d ${DATA_ROOT} ]; then
    echo "Erasing data root dir ${DATA_ROOT}"
    rm -rf "${DATA_ROOT}"
fi
mkdir "${DATA_ROOT}"

# test download scripts in parallel
./scripts/download_penobscot.sh "${DATA_PENOBSCOT}" & 
./scripts/download_dutch_f3.sh "${DATA_F3}" &

wait

# change imposed by download script
DATA_F3="${DATA_F3}/data"

# test preprocessing scripts
python scripts/prepare_penobscot.py split_inline --data-dir=${DATA_PENOBSCOT} --val-ratio=.1 --test-ratio=.2
python scripts/prepare_dutchf3.py split_train_val section --data-dir=${DATA_F3}
python scripts/prepare_dutchf3.py split_train_val patch --data-dir=${DATA_F3} --stride=50 --patch=100
