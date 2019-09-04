#!/bin/bash

echo "Downloading Dutch F3 from https://github.com/bolgebrygg/MalenoV"
# fetch Dutch F3 from Malenov project.
# wget https://drive.google.com/open?id=0B7brcf-eGK8CUUZKLXJURFNYeXM -O interpretation/voxel2pixel/F3/data.segy
wget https://github.com/waldeland/CNN-for-ASI/raw/master/F3/train/inline_339.png -O interpretation/voxel2pixel/F3/train/inline_339.png
wget https://github.com/waldeland/CNN-for-ASI/raw/master/F3/val/inline_405.png -O interpretation/voxel2pixel/F3/val/inline_405.png
echo "Download complete"
