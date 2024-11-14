#!/bin/bash

# Create and activate a new conda environment
conda create -n sd3_attack python=3.10 -y
conda init
conda activate sd3_attack

# # Install PyTorch with CUDA 11.8
# pip3 install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# # Install xformers for memory efficient attention
# pip install xformers==0.0.22.post4

# # Install triton for optimized CUDA kernels
# pip install triton==2.0.0

# Install other dependencies
pip install -r requirements.txt

# Verify CUDA installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('Device count:', torch.cuda.device_count())"