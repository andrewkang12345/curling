#!/usr/bin/env bash

# Shared runtime setup for local EC2 runs. Source this file after cd'ing to the
# csas_fixed_moreMCTS directory.
PYTHON="${PYTHON:-/opt/pytorch/bin/python}"

TEST_ENV_SITE="${TEST_ENV_SITE:-/mnt/data/curling2/testBrax/testEnv/lib/python3.12/site-packages}"
PYTORCH_SITE="${PYTORCH_SITE:-/opt/pytorch/lib/python3.12/site-packages}"

GPU_LIBS="${GPU_LIBS:-/opt/pytorch/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/opt/pytorch/lib/python3.12/site-packages/nvidia/cusparse/lib:/opt/pytorch/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/pytorch/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/pytorch/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/pytorch/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/opt/pytorch/lib/python3.12/site-packages/nvidia/cusolver/lib:/opt/pytorch/lib/python3.12/site-packages/nvidia/cufft/lib}"

export PYTHONPATH="$TEST_ENV_SITE:$PYTORCH_SITE:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$GPU_LIBS:${LD_LIBRARY_PATH:-}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.30}"
export CSAS_PRELOAD_NVIDIA_LIBS="${CSAS_PRELOAD_NVIDIA_LIBS:-1}"
