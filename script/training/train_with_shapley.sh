#!/bin/bash

# Set SwanLab environment variables
export SWANLAB_DISABLE_RICH=0
export SWANLAB_DISABLE_DEBUG=1
export SWANLAB_API_KEY=""  # SwanLab API key
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

# Network interface settings, use default interface
export NCCL_SOCKET_IFNAME=""

# Set CUDA related environment variables for performance optimization
export CUDA_DEVICE_MAX_CONNECTIONS=1     # Limit connections per device
export CUDA_LAUNCH_BLOCKING=0            # Disable CUDA launch blocking

# Ensure project directory is correct
PROJECT_DIR=$(pwd)
echo "Current working directory: $PROJECT_DIR"

# Check DeepSpeed configuration file
CONFIG_FILE="$PROJECT_DIR/src/config/accelerate_config/train_zero2.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: DeepSpeed configuration file does not exist: $CONFIG_FILE"
    exit 1
fi
echo "Using DeepSpeed configuration file: $CONFIG_FILE"

# Enable pure Token-level reward training!
echo "=========================================="
echo "🎯 Starting Pure Token-level Reward Training"
echo "=========================================="
echo "🏆 New Reward System (Max 4 points):"
echo "   🔹 Question tokens: Shapley reward(0-3) + Format reward(0-1)"
echo "   🔹 Answer tokens: Correctness reward(0-3) + Format reward(0-1)"
echo "   🔹 Other tokens: 0 points"
echo "📊 Format Reward Rules:"
echo "   ✅ Sentences starting with 'question:' → All tokens +1 point"
echo "   ✅ Sentences starting with 'answer:' → All tokens +1 point"
echo "🚀 Training Features:"
echo "   🎯 Token-level Group Baseline Advantage calculation"
echo "   📊 Each token's advantage = token_reward - group_baseline"
echo "   📈 SwanLab records mean rewards for each token type"
echo "   🎪 Precise token-level optimization"
echo "=========================================="

# Pure Token reward training configuration
USE_SHAPLEY=${USE_SHAPLEY:-"True"}              # Whether to use Shapley value weighting
ALPHA_REWARD=${ALPHA_REWARD:-"2.0"}             # Question Shapley reward weight
BETA_REWARD=${BETA_REWARD:-"1.0"}               # Question result reward weight
GAMMA_REWARD=${GAMMA_REWARD:-"3.0"}             # Answer correctness reward weight
FORMAT_REWARD_WEIGHT=${FORMAT_REWARD_WEIGHT:-"1.0"}  # Format reward weight

echo "🔧 Pure Token Reward Configuration:"
echo "   🧮 USE_SHAPLEY: $USE_SHAPLEY"
echo "   📝 ALPHA_REWARD: $ALPHA_REWARD (Question Shapley weight)"
echo "   🎯 BETA_REWARD: $BETA_REWARD (Question result weight)"
echo "   🏆 GAMMA_REWARD: $GAMMA_REWARD (Answer correctness weight)"
echo "   📋 FORMAT_REWARD_WEIGHT: $FORMAT_REWARD_WEIGHT (Format weight)"
echo "=========================================="

# Launch DeepSpeed ZeRO-2 configured pure Token reward training using accelerate
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file ./src/config/accelerate_config/train_zero2.yaml \
    --main_process_port 12349 \
    --num_processes 2 \
    --mixed_precision "bf16" \
    ./src/models/doctor_train.py \
    --use_token_level=True \
    --use_shapley="$USE_SHAPLEY" \
    --shapley_max_samples=50 \
    --shapley_min_samples=3 \
    --alpha_reward="$ALPHA_REWARD" \
    --beta_reward="$BETA_REWARD" \
    --gamma_reward="$GAMMA_REWARD" \
    --format_reward_weight="$FORMAT_REWARD_WEIGHT"