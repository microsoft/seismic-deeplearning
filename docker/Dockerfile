# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

FROM nvidia/cuda:10.2-base

# Install base dependencies
RUN apt-get update && \
    apt-get install -y curl git build-essential wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 

# Install Anaconda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
SHELL ["/bin/bash", "-c"] 

RUN wget --quiet https://repo.continuum.io/archive/Anaconda3-2019.10-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# download the seismic-deeplearning repo
RUN apt-get install -y zip && \
    wget --quiet https://github.com/microsoft/seismic-deeplearning/archive/master.zip -O master.zip && \
    unzip master.zip && rm master.zip

WORKDIR seismic-deeplearning-master

# RUN cd seismic-deeplearning-master && \
RUN conda update conda -y && \
    conda env create -n seismic-interpretation --file environment/anaconda/local/environment.yml && \
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

# TODO: In readme, ask user to download HRNet model and mount it to the docker container 
# TODO: maxkaz suggested we group RUN commands to reduce # of layers, and therefore, docker image size.  

# Run notebook
EXPOSE 8888/tcp

# CMD /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh" && \
#     conda run -n seismic-interpretation && \
CMD conda activate seismic-interpretation && \
    jupyter notebook --allow-root --ip 0.0.0.0 --port 8888 
# --notebook-dir seismic-deeplearning-master


