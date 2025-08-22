#!/bin/bash
# Wrapper script for training with proper environment setup

# Configure environment variables
export CONDA_PREFIX="${CONDA_PREFIX:-/home/bernardorozco/miniconda3/envs/robo-poet-gpu}"
export CUDA_HOME="$CONDA_PREFIX"
export TF_CPP_MIN_LOG_LEVEL="2"
export CUDA_VISIBLE_DEVICES="0"
export TF_FORCE_GPU_ALLOW_GROWTH="true"

# Configure LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH"

# Execute training with proper Python interpreter
exec "$CONDA_PREFIX/bin/python" robo_poet.py "$@"