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

RUN wget --quiet https://repo.continuum.io/archive/Anaconda3-2019.10-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
    

# download the seismic-deeplearning repo
WORKDIR /root
RUN apt-get install -y zip && \
    wget --quiet https://github.com/microsoft/seismic-deeplearning/archive/master.zip -O master.zip && \
    unzip master.zip && rm master.zip 

# Setup the environment: 
RUN cd seismic-deeplearning-master && \
    conda env create -f environment/anaconda/local/environment.yml && \
    echo "conda activate seismic-interpretation" >> ~/.bashrc && \
    conda activate seismic-interpretation && \
    pip install -e interpretation && \
    pip install -e cv_lib


ENV PATH="${HOME}/conda/envs/seismic-deeplearning/bin:${PATH}"



# Copy data : #TODO
# TODO: In readme, ask user to download HRNet model and mount it to the docker container 

# VOLUME 

# Run notebook
EXPOSE 8888/tcp
# CMD ["conda", "run", "-n", "seismic-interpretation", "jupyter", "notebook", "--allow-root", "--ip", "0.0.0.0", "--port", "8888"]
# CMD jupyter notebook --allow-root --ip 0.0.0.0 --port 8888 --no-browser --notebook-dir .
CMD source /data/anaconda/etc/profile.d/conda.sh && \
    conda activate seismic-interpretation && \
    jupyter notebook --allow-root --ip 0.0.0.0 --port 8888 

# command to run, and map ports:
# sudo docker run --rm -it -p 8888:8888 seismic-deeplearning