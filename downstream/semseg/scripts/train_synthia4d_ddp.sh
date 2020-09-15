#!/bin/bash

set -x
# Exit script when a command returns nonzero state
set -e

set -o pipefail

SAVEPATH=/checkpoint/s9xie/space/3d_ssl2

MODEL=Res16UNet34C

export PYTHONUNBUFFERED="True"
# export CUDA_VISIBLE_DEVICES=$1

export BATCH_SIZE=${BATCH_SIZE:-9}
export DATASET=${DATASET:-SynthiaCVPR15cmVoxelizationDataset}
# export MODEL=${MODEL:-Res16UNet14A}

export TIME=$(date +"%Y-%m-%d_%H-%M-%S-%N")

export LOG_DIR=${SAVEPATH}/outputs/Synthia4D/slurm/SSL_$MODEL_$TIME

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR
LOG="$LOG_DIR/$TIME.txt"

python -m ddp_main \
    --log_dir $LOG_DIR \
    --batch_size $BATCH_SIZE \
    --scheduler PolyLR \
    --model $MODEL \
    --conv1_kernel_size 3 \
    --lr ${LR:-0.1} \
    --dataset $DATASET \
    --max_iter 15000 \
    --stat_freq 5 \
    --test_stat_freq 100 \
    --save_freq 125 \
    --val_freq 125 \
    --train_limit_numpoints 1200000 \
    --train_phase train \
    --synthia_path /private/home/s9xie/jiatao/SpatioTemporalSegmentation/synthia4d \
    $3 2>&1 | tee -a "$LOG"

python -m main \
    --log_dir $LOG_DIR \
    --dataset $DATASET \
    --model $MODEL \
    --is_train False \
    --weights $LOG_DIR/weights.pth \
    $3 2>&1 | tee -a "$LOG"
