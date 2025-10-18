#!/bin/bash

# Set SwanLab environment variables
export SWANLAB_DISABLE_RICH=0
export SWANLAB_DISABLE_DEBUG=1
export SWANLAB_API_KEY=""
export SWANLAB_ENTITY=""

# Set DeepSpeed and distributed environment variables
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL

# NCCL optimization settings
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=8
export NCCL_SHM_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_BUFFSIZE=4194304
export NCCL_CROSS_NIC=0
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_SOCKET_IFNAME=""

# Set CUDA related environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=0

# Ensure project directory is correct
PROJECT_DIR=$(pwd)
echo "Current working directory: $PROJECT_DIR"

# Check configuration file
CONFIG_FILE="$PROJECT_DIR/src/config/accelerate_config/train_zero2.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file does not exist: $CONFIG_FILE"
    exit 1
fi

echo "=========================================="
echo "🎯 Starting Rollout-level Shapley Value Training"
echo "=========================================="
echo "🏆 Training Mode: Rollout-level"
echo "   🔹 Using overall_reward function"
echo "   🔹 Enable Shapley value weighted fact scores"
echo "   🔹 Each rollout receives a comprehensive score"
echo "   🔹 Group Baseline calculated at rollout level"
echo "=========================================="

# Simplified training configuration
USE_SHAPLEY="True"
USE_TOKEN_LEVEL="False"

echo "🔧 Training Configuration:"
echo "   🧮 USE_SHAPLEY: $USE_SHAPLEY"
echo "   📏 USE_TOKEN_LEVEL: $USE_TOKEN_LEVEL"
echo "=========================================="

# Launch training
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --config_file ./src/config/accelerate_config/train_zero2.yaml \
    --main_process_port 12349 \
    --num_processes 1 \
    --mixed_precision "bf16" \
    ./src/models/doctor_train.py \
    --use_token_level="$USE_TOKEN_LEVEL" \
    --use_shapley="$USE_SHAPLEY"