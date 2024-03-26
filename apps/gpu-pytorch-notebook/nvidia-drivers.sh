#!/bin/bash

#############################
## Check if GPU is Present ##
#############################

#cat /proc/driver/nvidia/version
echo "Installing Nvidia Drivers"

if [ $? -eq 0 ]; then
    echo "GPU Detected!"

    # Removing old apt-key
    apt-key del 7fa2af80

    # Dowloading and installing nvidia keyring
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb

    # Getting latest repo updates for nvidia
    apt-get update

    # Installing sudo and dependencies for nvidia drivers
    apt-get install -y sudo linux-modules-nvidia-470-gcp

    # Installing packages and drivers
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-driver-470

    # Downloading and installing compatible Cuda drivers
    wget --no-verbose --progress=bar https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/cuda_11.4.4_470.82.01_linux.run
    sh cuda_11.4.4_470.82.01_linux.run --silent --toolkit

    # Setting PATH for cuda
    export PATH=/usr/local/cuda-11.4/bin${PATH:+:${PATH}}
    export CUDA_HOME=/usr/local/cuda

    # Cleaning packages and downloads
    apt clean
    rm -rf /var/cache/apt/archives/*
    rm -rf cuda-keyring_1.1-1_all.deb
    rm -rf cuda_11.4.4_470.82.01_linux.run

    # Testing installation
    nvcc --version
else
    echo "No GPU Detected..."
fi