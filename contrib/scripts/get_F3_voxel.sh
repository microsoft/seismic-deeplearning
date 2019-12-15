#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

echo "Make sure you also download Dutch F3 data from https://github.com/bolgebrygg/MalenoV"
# fetch Dutch F3 from Malenov project.
# wget https://drive.google.com/open?id=0B7brcf-eGK8CUUZKLXJURFNYeXM -O interpretation/voxel2pixel/F3/data.segy

if [ $# -eq 0 ]
then
    downdirtrain='experiments/interpretation/voxel2pixel/F3/train'
    downdirval='experiments/interpretation/voxel2pixel/F3/val'
else
    downdirtrain=$1
    downdirval=$1
fi

mkdir -p ${downdirtrain}
mkdir -p ${downdirval}

echo "Downloading train label to $downdirtrain and validation label to $downdirval"
wget https://github.com/waldeland/CNN-for-ASI/raw/master/F3/train/inline_339.png -O ${downdirtrain}/inline_339.png
wget https://github.com/waldeland/CNN-for-ASI/raw/master/F3/val/inline_405.png -O ${downdirval}/inline_405.png
echo "Download complete"
