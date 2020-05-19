# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Pull from cuda 10.2 image (based on Ubuntu 18.04): 
FROM nvidia/cuda:10.2-base

# Install base dependencies
RUN apt-get update && \
    apt-get install -y curl git build-essential wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
SHELL ["/bin/bash", "-c"] 

# Install Anaconda and download the seismic-deeplearning repo
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    apt-get install -y zip && \
    wget --quiet https://github.com/microsoft/seismic-deeplearning/archive/staging.zip -O staging.zip && \
    unzip staging.zip && rm staging.zip

RUN cd seismic-deeplearning-staging && \
    conda env create -n seismic-interpretation --file environment/anaconda/local/environment.yml && \
    source activate seismic-interpretation && \
    python -m ipykernel install --user --name seismic-interpretation && \
    pip install -e interpretation && \
    pip install -e cv_lib 

# TODO: add back in later when Penobscot notebook is available
# Download Penobscot dataset:
# RUN cd seismic-deeplearning-staging && \
#    data_dir="/home/username/data/penobscot" && \
#    mkdir -p "$data_dir" && \
#    ./scripts/download_penobscot.sh "$data_dir"  && \
#    cd scripts  && \
#    source activate seismic-interpretation && \
#    python prepare_penobscot.py split_inline --data-dir=$data_dir --val-ratio=.1 --test-ratio=.2  && \
#    cd ..

# Download F3 dataset:
RUN cd seismic-deeplearning-staging && \
    data_dir="/home/username/data/dutch" && \
    mkdir -p "$data_dir" && \
    ./scripts/download_dutch_f3.sh "$data_dir" && \
    cd scripts && \
    source activate seismic-interpretation && \
    python prepare_dutchf3.py split_train_val section --data-dir=${data_dir}/data --label_file=train/train_labels.npy --output_dir=splits --split_direction=both && \
    python prepare_dutchf3.py split_train_val patch --data-dir=${data_dir}/data --label_file=train/train_labels.npy --output_dir=splits --stride=50 --patch_size=100 --split_direction=both && \
    cd ..

# Run notebook
EXPOSE 9000/tcp
# TensorBoard inside notebook
EXPOSE 9001/tcp

CMD source activate seismic-interpretation && \
    jupyter lab --allow-root --ip 0.0.0.0 --port 9000
