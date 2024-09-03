FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

# Add the deadsnakes PPA for newer Python versions
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa

# Update the system and install dependencies including Python 3.11
RUN apt-get update && apt-get install -y \
    python3-opencv ca-certificates python3.11 python3.11-dev git wget sudo ninja-build

# Link python3.11 to python3 for ease of use
RUN ln -sfv /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sfv /usr/bin/python3.11 /usr/bin/python

# Create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system --uid ${USER_ID} appuser -g sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER appuser
WORKDIR /home/appuser

# Add local bin to PATH
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Install pip for Python 3.11
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py --user && \
    rm get-pip.py

# Install PyTorch and other dependencies via pip
# Make sure the correct wheel URL or index URL that supports Python 3.11 is used
RUN pip install --user tensorboard cmake onnx
RUN pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install additional Python packages
RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'

# Clone and install Detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
RUN pip install --user -e detectron2_repo	

RUN pip install git+https://github.com/cocodataset/panopticapi.git
RUN pip install opencv-python

# Set a fixed model cache directory
ENV FVCORE_CACHE="/tmp"
WORKDIR /home/appuser/detectron2_repo