#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation. All rights reserved. 
# Licensed under the MIT License. 
# commitHash: 9abd6c82b2319ea5bfa9ff1c9a56478caad07ab9
# url: https://github.com/yalaudah/facies_classification_benchmark
#
# Download the Dutch F3 dataset and extract
if [ ! -d $1 ]; then 
    echo "$1 does not exist"
    exit 1;
fi;

echo "Extracting to $1"
# Download the files:  
temp_file=$(mktemp -d)/data.zip
wget -o /dev/null -O $temp_file https://zenodo.org/record/3755060/files/data.zip

# Check that the md5 checksum matches to varify file integrity
echo "Expected output: MD5(data.zip)= bc5932279831a95c0b244fd765376d85"

openssl dgst -md5 $temp_file
 
# Unzip the data 
unzip -d $1 $temp_file && rm $temp_file
mkdir $1/data/splits
echo "Download complete"
