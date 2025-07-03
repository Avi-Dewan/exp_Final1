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
echo "Config file: $CONFIG_FILE"
echo "Log file: $LOG_FILE"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    echo "Available files in configurations directory:"
    ls -la "$SCRIPT_DIR/configurations/" || echo "Configurations directory not found"
    exit 1
fi

echo "Config file found, reading configurations..."

# Training parameters
COMMON_ARGS="--pretrain_dir $PRETRAIN_DIR --DTC sinkhornEnhanced_softBCE --topk 25 --warmup_epochs 1 --epochs 2 --rampup_length_softBCE 5 --rampup_coefficient_softBCE 10 --log_file $LOG_FILE"

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
    
    # Debug: Show the full command being executed
    echo "Executing command:"
    echo "CUDA_VISIBLE_DEVICES=0 python3 $PROJECT_DIR/gcd.py $COMMON_ARGS --imbalance_config \"$config\" --config_name \"$config_name\""
    
    CUDA_VISIBLE_DEVICES=0 python3 "$PROJECT_DIR/gcd.py" $COMMON_ARGS --imbalance_config "$config" --config_name "$config_name"
    
    echo "Completed: $config_name"
    echo ""
}

# Main execution
echo ""
echo "Extracting configurations..."

# Use Python to process configs and run training directly
python3 -c "
import json
import sys
import os
import subprocess

# Read config file
config_file = '$CONFIG_FILE'
start_idx = int('$START_INDEX')
end_idx = int('$END_INDEX')
project_dir = '$PROJECT_DIR'
common_args = '$COMMON_ARGS'

print(f'Reading config file: {config_file}')

try:
    with open(config_file, 'r') as f:
        configs = json.load(f)
    
    total_configs = len(configs)
    print(f'Total configs available: {total_configs}')
    
    # Validate range
    if start_idx < 0 or start_idx >= total_configs or end_idx < 0 or end_idx >= total_configs or start_idx > end_idx:
        print(f'Error: Invalid range {start_idx}-{end_idx} for {total_configs} configs')
        sys.exit(1)
    
    # Select configs to run
    selected_configs = configs[start_idx:end_idx+1]
    print(f'Selected {len(selected_configs)} configs to run')
    
    # Run each config
    for i, config_data in enumerate(selected_configs):
        actual_index = start_idx + i
        config_name = config_data['name']
        config_value = config_data['config']
        
        print(f'\\n[{i+1}/{len(selected_configs)}] Processing config {actual_index}: {config_name}')
        print(f'Config value: {config_value}')
        
        # Build command
        cmd = [
            'python3', f'{project_dir}/gcd.py'
        ] + common_args.split() + [
            '--imbalance_config', config_value,
            '--config_name', config_name
        ]
        
        print(f'Running command: {\" \".join(cmd)}')
        
        # Set environment and run
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        
        try:
            result = subprocess.run(cmd, env=env, check=True, capture_output=False)
            print(f'Completed config {actual_index}: {config_name}')
        except subprocess.CalledProcessError as e:
            print(f'Error running config {actual_index}: {config_name}')
            print(f'Return code: {e.returncode}')
            # Continue with next config instead of stopping
            continue
        except Exception as e:
            print(f'Unexpected error running config {actual_index}: {e}')
            continue
    
    print('\\n' + '='*60)
    print('All configurations completed!')
    print('='*60)

except FileNotFoundError:
    print(f'Error: Config file not found: {config_file}')
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f'Error parsing JSON config file: {e}')
    sys.exit(1)
except Exception as e:
    print(f'Unexpected error: {e}')
    sys.exit(1)
"

echo "=========================================="
echo "All training completed!"
echo "Results saved to: $LOG_FILE"
echo "=========================================="