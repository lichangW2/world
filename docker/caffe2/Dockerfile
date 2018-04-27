FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04
LABEL maintainer="lichangW@cj.com"

# caffe2 install with gpu support

RUN sed -i s/archive.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
RUN sed -i s/security.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
# 这两个 NVIDIA source list 更新存在问题
RUN rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgflags-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libiomp-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libopenmpi-dev \
    libprotobuf-dev \
    libsnappy-dev \
    openmpi-bin \
    openmpi-doc \
    protobuf-compiler \
    python-dev \
    python-numpy \
    python-pydot \
    python-setuptools \
    python-scipy \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -k https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    flask \
    future \
    graphviz \
    hypothesis \
    jupyter \
    matplotlib \
    numpy \
    protobuf \
    pydot \
    python-nvd3 \
    pyyaml \
    requests \
    scikit-image \
    scipy \
    setuptools \
    six \
    tornado

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    python-software-properties \
    ca-certificates  \
    vim \
    lrzsz \
    unzip
########## INSTALLATION STEPS ###################
#RUN git clone --branch master --recursive  https://github.com/caffe2/caffe2.git
#RUN git clone --recursive https://github.com/pytorch/pytorch.git && cd pytorch && \
#    git submodule update --init
RUN mkdir /pytorch
ADD pytorch.tar /pytorch/
RUN cd /pytorch && mkdir build && cd build && \
     cmake .. \
     -DCUDA_ARCH_NAME=Manual \
     -DCUDA_ARCH_BIN="35 50 52" \
     -DCUDA_ARCH_PTX="52" \
     && make -j"$(nproc)" install \
     && ldconfig \
     && make clean \
     && cd .. \
     && rm -rf build
ENV PYTHONPATH /usr/local
