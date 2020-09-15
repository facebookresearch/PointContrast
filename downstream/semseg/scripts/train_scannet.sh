#!/bin/bash

set -x
# Exit script when a command returns nonzero state
set -e

set -o pipefail

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

export BATCH_SIZE=${BATCH_SIZE:-9}

export TIME=$(date +"%Y-%m-%d_%H-%M-%S")

export LOG_DIR=./outputs/ScanNet$2/$TIME

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR

LOG="$LOG_DIR/$TIME.txt"

python -m main \
    --log_dir $LOG_DIR \
    --dataset ScannetVoxelization2cmDataset \
    --model Res16UNet34C \
    --lr 1e-1 \
    --batch_size $BATCH_SIZE \
    --scheduler PolyLR \
    --max_iter 120000 \
    --train_limit_numpoints 1200000 \
    --train_phase train \
    $3 2>&1 | tee -a "$LOG"

export TIME=$(date +"%Y-%m-%d_%H-%M-%S")
LOG="$LOG_DIR/$TIME.txt"

python -m main \
    --log_dir $LOG_DIR \
    --dataset ScannetVoxelization2cmDataset \
    --model Res16UNet34C \
    --lr 1e-2 \
    --batch_size $BATCH_SIZE \
    --scheduler PolyLR \
    --max_iter 120000 \
    --train_limit_numpoints 1200000 \
    --train_phase trainval \
    --weights $LOG_DIR/weights.pth \
    $3 2>&1 | tee -a "$LOG"
