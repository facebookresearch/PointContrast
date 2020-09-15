#!/bin/bash

set -x
# Exit script when a command returns nonzero state
set -e

set -o pipefail

SAVEPATH=/checkpoint/s9xie/space/3d_ssl2
MODEL=Res16UNet34C

export PYTHONUNBUFFERED="True"
# export CUDA_VISIBLE_DEVICES=0
export BATCH_SIZE=${BATCH_SIZE:-6}
export TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export LOG_DIR=${SAVEPATH}/outputs/ScanNetDataset/slurm/$MODEL_$TIME

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR
LOG="$LOG_DIR/$TIME.txt"

python -m ddp_main \
    --dataset ScannetVoxelization2cmDataset \
    --batch_size $BATCH_SIZE \
    --scheduler PolyLR \
    --model ${MODEL} \
    --conv1_kernel_size 3 \
    --log_dir $LOG_DIR \
    --lr 0.1 \
    --max_iter 15000 \
    --stat_freq 5 \
    --test_stat_freq 100 \
    --save_freq 125 \
    --val_freq 125 \
    --train_limit_numpoints 1200000 \
    --train_phase train \
    --scannet_path /private/home/s9xie/3d_ssl/SpatioTemporalSegmentation/data 2>&1 | tee -a "$LOG"
