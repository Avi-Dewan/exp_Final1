#!/bin/bash

# Kaggle CIFAR-10 Training Script
# Usage: ./cifar10_script.sh <pretrain_dir> [start_index] [end_index] [config_file]

set -e

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <pretrain_dir> [start_index] [end_index] [config_file]"
    echo "Example: $0 /kaggle/input/simclr_finetuned_cifar-10/pytorch/default/1/resnet_simCLR_finetuned.pth"
    echo "Example: $0 /kaggle/input/simclr_finetuned_cifar-10/pytorch/default/1/resnet_simCLR_finetuned.pth 0 3"
    echo "Example: $0 /kaggle/input/simclr_finetuned_cifar-10/pytorch/default/1/resnet_simCLR_finetuned.pth 0 3 severe_imbalance_configs.json"
    exit 1
fi

PRETRAIN_DIR="$1"
START_INDEX="${2:-0}"
END_INDEX="${3:-7}"  # Default to all 8 configs (0-7)
CONFIG_FILENAME="${4:-imbalance_configs.json}"  # Default config file

# Check pretrained file exists
if [ ! -f "$PRETRAIN_DIR" ]; then
    echo "Error: Pretrain file not found at $PRETRAIN_DIR"
    exit 1
fi

# Get script directory and set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$SCRIPT_DIR/configurations/$CONFIG_FILENAME"
LOG_FILE="/kaggle/working/training_results.csv"

echo "Starting CIFAR-10 training..."
echo "Pretrain file: $PRETRAIN_DIR"
echo "Running configs: $START_INDEX to $END_INDEX"
echo "Log file: $LOG_FILE"

# Extract configs from JSON file
extract_configs() {
    python3 -c "
import json
import sys

with open('$CONFIG_FILE', 'r') as f:
    configs = json.load(f)

total_configs = len(configs)
start_idx = int('$START_INDEX')
end_idx = int('$END_INDEX')

# Validate range
if start_idx < 0 or start_idx >= total_configs or end_idx < 0 or end_idx >= total_configs or start_idx > end_idx:
    print(f'Error: Invalid range {start_idx}-{end_idx} for {total_configs} configs', file=sys.stderr)
    sys.exit(1)

# Output selected configs
for i in range(start_idx, end_idx + 1):
    config = configs[i]
    print(f'{i}|||{config[\"name\"]}|||{config[\"config\"]}')
"
}

# Training parameters
COMMON_ARGS="--pretrain_dir $PRETRAIN_DIR --DTC sinkhornEnhanced_softBCE --topk 25 --warmup_epochs 10 --epochs 200 --rampup_length_softBCE 5 --rampup_coefficient_softBCE 10 --log_file $LOG_FILE"

# Run training
run_training() {
    local config="$1"
    local config_name="$2"
    local index="$3"
    
    echo ""
    echo "=========================================="
    echo "Running config $index: $config_name"
    echo "Config: $config"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=0 python3 "$PROJECT_DIR/gcd.py" $COMMON_ARGS --imbalance_config "$config" --config_name "$config_name"
    
    echo "Completed: $config_name"
    echo ""
}

# Main execution
echo ""
echo "Extracting configurations..."

# Get configurations
CONFIG_OUTPUT=$(extract_configs)

# Count configs to run
CONFIG_COUNT=$(echo "$CONFIG_OUTPUT" | wc -l)
echo "Will run $CONFIG_COUNT configurations"

# Process each config
config_counter=1
echo "$CONFIG_OUTPUT" | while IFS='|||' read -r index config_name config_value; do
    echo "[$config_counter/$CONFIG_COUNT] Processing config $index..."
    run_training "$config_value" "$config_name" "$index"
    config_counter=$((config_counter + 1))
done

echo "=========================================="
echo "All training completed!"
echo "Results saved to: $LOG_FILE"
echo "=========================================="