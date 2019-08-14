# Taken from https://github.com/halhenke/docker-sniper/blob/master/docker-master/Dockerfile

#FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
FROM nvcr.io/nvidia/cuda:9.0-cudnn7.2-devel-ubuntu16.04

USER root

RUN apt-get update \
  && apt-get -y install wget locales git bzip2 curl python-pip \
  && rm -rf /var/lib/apt/lists/*

RUN localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
  && locale-gen en_US.utf8 \
  && /usr/sbin/update-locale LANG=en_US.UTF-8

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8%

WORKDIR /root

ENV LD_LIBRARY_PATH /usr/local/cuda/lib64

# Install sudo
RUN apt-get update && \
  apt-get -y install sudo \
  && rm -rf /var/lib/apt/lists/*

RUN useradd -m docker \
  && echo "docker:docker" | chpasswd && adduser docker sudo

RUN git clone --recursive git@git.iai.local:ash95/SNIPER-mxnet.git #https://github.com/ash1995/SNIPSAT.git
#RUN git clone --recursive https://git.iai.local/ash95/sniper.git

WORKDIR /root/SNIPER-mxnet

# * apt-get install software-properties-common
RUN apt-get update && \
    apt-get -y install  \
    software-properties-common \
    libatlas-base-dev \
    libopencv-dev \
    libopenblas-dev \
    libgdal-dev \
    gcc-5 \
    g++-5 \
  && rm -rf /var/lib/apt/lists/*
  
RUN make USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
# RUN make -j 28 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
# RUN make -j [NUM_OF_PROCESS] USE_CUDA_PATH=[PATH_TO_THE_CUDA_FOLDER] USE_NCCL=1 USE_NCCL_PATH=[PATH_TO_NCCL_INSTALLATION]

RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal 
RUN export C_INCLUDE_PATH=/usr/include/gdal

RUN apt-get update && apt-get -y install python-gdal

# Service stuff
RUN add-apt-repository -y ppa:ubuntugis/ppa && \
    apt update && apt -y upgrade && apt install python3-rtree

COPY requirements.txt /tmp/requirements.txt

RUN apt-get update && pip install --upgrade pip && pip2 install -r requirements.txt

WORKDIR /sniper
