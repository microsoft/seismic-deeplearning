#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation. All rights reserved. 
# Licensed under the MIT License. 
# commitHash: 
# url: https://zenodo.org/record/1341774
#
# Download the penobscot dataset and extract
# Files description
# File 	    Format 	                    Num Files 	    Total size (MB)
# H1-H7 	XYZ 	                         7 	             87.5
# Seismic   inlines 	TIF 	           601 	            1,700
# Seismic   crosslines 	TIF 	           481 	            1,700
# Labeled   inlines 	PNG 	           601 	              4.9
# Labeled   crosslines 	PNG 	           481 	              3.9
# Seismic   tiles (train) 	PNG 	    75,810 	              116
# Seismic   labels (train) 	JSON 	         2 	              1.5
# Seismic   tiles (test) 	PNG 	    28,000 	              116
# Seismic   labels (test) 	JSON 	         2 	              0.5
# Args: directory to download and extract data to
# Example: ./download_penobscot.sh /mnt/penobscot


echo Extracting to $1 
cd $1
# Download the files:  
wget https://zenodo.org/record/1341774/files/crosslines.zip
wget https://zenodo.org/record/1341774/files/inlines.zip
wget https://zenodo.org/record/1341774/files/horizons.zip
wget https://zenodo.org/record/1341774/files/masks.zip
wget https://zenodo.org/record/1341774/files/tiles_crosslines.zip
wget https://zenodo.org/record/1341774/files/tiles_inlines.zip 
 
# Check that the md5 checksum matches to varify file integrity
# 
# Expected output:
# MD5(crosslines.zip)= 7bbe432052fe41c6009d9437fd0929b8
# MD5(horizons.zip)= 42c104fafbb8e79695ae23527a91ee78
# MD5(inlines.zip)= 0553676ef48879f590378cafc12d165d
# MD5(masks.zip)= 12f142cb33af55c3b447401ebd81aba1
# MD5(tiles_crosslines.zip)= 8dbd99da742ac9c6f9b63f8c6f925f6d
# MD5(tiles_inlines.zip)= 955e2f9afb01878df2f71f0074736e42

openssl dgst -md5 crosslines.zip
openssl dgst -md5 horizons.zip
openssl dgst -md5 inlines.zip   
openssl dgst -md5 masks.zip   
openssl dgst -md5 tiles_crosslines.zip   
openssl dgst -md5 tiles_inlines.zip  
 
# Unzip the data 
unzip crosslines.zip
unzip inlines.zip
unzip horizons.zip
unzip masks.zip
unzip tiles_crosslines.zip
unzip tiles_inlines.zip

echo Download complete.