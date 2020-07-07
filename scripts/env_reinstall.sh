#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

conda env remove -n seismic-interpretation
yes | conda env create -f environment/anaconda/local/environment.yml
# don't use conda here as build VM's shell isn't setup when running as a build agent
source activate seismic-interpretation
pip install -e cv_lib
pip install -e interpretation
# temporary DS VM bugfix
yes | conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
