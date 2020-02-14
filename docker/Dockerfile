# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

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

WORKDIR seismic-deeplearning-staging

RUN conda env create -n seismic-interpretation --file environment/anaconda/local/environment.yml && \
    source activate seismic-interpretation && \
    pip install -e interpretation && \
    pip install -e cv_lib 

# Download Penobscot dataset:
RUN data_dir="../data/penobscot" && \
    mkdir -p "$data_dir" && \
    ./scripts/download_penobscot.sh "$data_dir"

# Download F3 dataset:
RUN data_dir="..//data/dutch" && \
    mkdir -p "$data_dir" && \
    ./scripts/download_dutch_f3.sh "$data_dir"

# To prepare the data for the experiments (run prepare_ scripts)

# Run notebook
EXPOSE 8888/tcp

CMD source activate seismic-interpretation && \
    python -m ipykernel install --user --name seismic-interpretation && \
    jupyter notebook --allow-root --ip 0.0.0.0 --port 8888 


