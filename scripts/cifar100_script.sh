#!/bin/bash

# Kaggle CIFAR-100 Training Script
# Usage: ./cifar100_script.sh <pretrain_dir> <config_file> <start_index> <end_index> [warmup_epochs] [epochs]

set -e

# Check arguments
if [ $# -lt 4 ]; then
    echo "Usage: $0 <pretrain_dir> <config_file> <start_index> <end_index> [warmup_epochs] [epochs]"
    echo "Example: $0 /kaggle/input/resnet_simclr_finetuned_cifar100/pytorch/default/1/resnet_simCLR_finetuned.pth cifar_100_imbalance_config.json 0 1"
    echo "Example: $0 /kaggle/input/resnet_simclr_finetuned_cifar100/pytorch/default/1/resnet_simCLR_finetuned.pth cifar_100_imbalance_config.json 0 1 20 300"
    exit 1
fi

PRETRAIN_DIR="$1"
CONFIG_FILENAME="$2"
START_INDEX="$3"
END_INDEX="$4"
WARMUP_EPOCHS="${5:-10}"  # Default to 10
EPOCHS="${6:-200}"        # Default to 200

# Check pretrained file exists
if [ ! -f "$PRETRAIN_DIR" ]; then
    echo "Error: Pretrain file not found at $PRETRAIN_DIR"
    exit 1
fi

# Get script directory and set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$SCRIPT_DIR/configurations/$CONFIG_FILENAME"
LOG_FILE="/kaggle/working/training_results_cifar100.csv"

echo "Starting CIFAR-100 training..."
echo "Pretrain file: $PRETRAIN_DIR"
echo "Config file: $CONFIG_FILENAME"
echo "Running configs: $START_INDEX to $END_INDEX"
echo "Warmup epochs: $WARMUP_EPOCHS, Training epochs: $EPOCHS"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

# Training parameters (CIFAR-100 specific)
COMMON_ARGS="--dataset_name cifar100 --n_labeled_classes 80 --n_unlabeled_classes 20 --pretrain_dir $PRETRAIN_DIR --topk 25 --warmup_epochs $WARMUP_EPOCHS --epochs $EPOCHS --rampup_length_softBCE 5 --rampup_coefficient_softBCE 10 --log_file $LOG_FILE"

# Main execution
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

try:
    with open(config_file, 'r') as f:
        configs = json.load(f)
    
    total_configs = len(configs)
    
    # Validate range
    if start_idx < 0 or start_idx >= total_configs or end_idx < 0 or end_idx >= total_configs or start_idx > end_idx:
        print(f'Error: Invalid range {start_idx}-{end_idx} for {total_configs} configs')
        sys.exit(1)
    
    # Select configs to run
    selected_configs = configs[start_idx:end_idx+1]
    print(f'Will run {len(selected_configs)} configurations')
    
    # Run each config
    for i, config_data in enumerate(selected_configs):
        actual_index = start_idx + i
        config_name = config_data['name']
        config_value = config_data['config']
        
        print(f'\\n[{i+1}/{len(selected_configs)}] Running config {actual_index}: {config_name}')
        
        # Build command
        cmd = [
            'python3', f'{project_dir}/gcd.py'
        ] + common_args.split() + [
            '--imbalance_config', config_value,
            '--config_name', config_name
        ]
        
        # Set environment and run
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        
        try:
            subprocess.run(cmd, env=env, check=True)
            print(f'Completed: {config_name}')

        except subprocess.CalledProcessError:
            print(f'Error running config {actual_index}: {config_name}')
            continue

        except Exception as e:
            print(f'Unexpected error: {e}')
            continue
    
    print('\\n' + '='*50)
    print('All configurations completed!')
    print('='*50)

except FileNotFoundError:
    print(f'Error: Config file not found: {config_file}')
    sys.exit(1)

except json.JSONDecodeError as e:
    print(f'Error parsing JSON config file: {e}')
    sys.exit(1)
    
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
"

echo "Results saved to: $LOG_FILE"