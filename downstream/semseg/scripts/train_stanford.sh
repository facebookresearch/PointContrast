#!/bin/bash

set -x
# Exit script when a command returns nonzero state
set -e

set -o pipefail

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1
export BATCH_SIZE=${BATCH_SIZE:-6}
export TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export LOG_DIR=./outputs/StanfordArea5Dataset/$2/$TIME

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR

LOG="$LOG_DIR/$TIME.txt"

python -m main \
    --dataset StanfordArea5Dataset \
    --batch_size $BATCH_SIZE \
    --scheduler PolyLR \
    --model Res16UNet18 \
    --conv1_kernel_size 5 \
    --log_dir $LOG_DIR \
    --lr 1e-1 \
    --max_iter 60000 \
    --data_aug_color_trans_ratio 0.05 \
    --data_aug_color_jitter_std 0.005 \
    --train_phase train \
    $3 2>&1 | tee -a "$LOG"
