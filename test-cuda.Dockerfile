FROM nvidia/cuda:9.2-base-ubuntu16.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment
RUN /home/user/miniconda/bin/conda create -y --name py36 python=3.6.9 \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
RUN /home/user/miniconda/bin/conda install conda-build=3.18.9=py36_3 \
 && /home/user/miniconda/bin/conda clean -ya

# CUDA 9.2-specific steps
RUN conda install -y -c pytorch \
    cudatoolkit=9.2 \
    "pytorch=1.4.0=py3.6_cuda9.2.148_cudnn7.6.3_0" \
    "torchvision=0.5.0=py36_cu92" \
 && conda clean -ya

# Install HDF5 Python bindings
RUN conda install -y h5py=2.8.0 \
 && conda clean -ya
RUN pip install h5py-cache==1.0

# Install Torchnet, a high-level framework for PyTorch
RUN pip install torchnet==0.0.4

# Install Requests, a Python library for making HTTP requests
RUN conda install -y requests=2.19.1 \
 && conda clean -ya

# Install Graphviz
RUN conda install -y graphviz=2.40.1 python-graphviz=0.8.4 \
 && conda clean -ya

# Install OpenCV3 Python bindings
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    libgtk2.0-0 \
    libcanberra-gtk-module \
 && sudo rm -rf /var/lib/apt/lists/*
RUN conda install -y -c menpo opencv3=3.1.0 \
 && conda clean -ya
MAINTAINER Valentin Dittmar mail@valentindittmar.eu

RUN add-apt-repository -r ppa:jonathonf/python-3.6 -y
#RUN apt-key update -y
RUN apt update
#RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update -y
RUN apt-get install -y --no-install-recommends apt-utils zsh make wget nano curl unzip -y
RUN apt-get install gcc g++ libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev -y
RUN apt-get install python3 python3-dev python3-dev build-essential python-pip -y
RUN apt-get install python3.6 python3.6-dev python3.6-dev build-essential python-pip -y

RUN apt-get upgrade python3.6 -y
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2
RUN wget https://bootstrap.pypa.io/get-pip.py --no-check-certificate
RUN python3 get-pip.py

RUN python3 -V
RUN pip3 -V
RUN pip3 install --upgrade pip
RUN pip3 install -U setuptools
RUN pip3 install torch tensorflow
RUN python3 -c "import torch;print(torch.cuda.is_available())"
RUN python3 -c "import tensorflow as tf;print(tf.config.experimental.list_physical_devices())"
