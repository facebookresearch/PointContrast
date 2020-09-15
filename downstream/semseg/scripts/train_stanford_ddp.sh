#!/bin/bash

set -x
# Exit script when a command returns nonzero state
set -e

set -o pipefail

SAVEPATH=/checkpoint/s9xie/space/3d_ssl2
DATAPATH=${3:-"/private/home/s9xie/data/3d_ssl2/Stanford3D/"}
MODEL=Res16UNet34C

export PYTHONUNBUFFERED="True"
# export CUDA_VISIBLE_DEVICES=0
export BATCH_SIZE=${BATCH_SIZE:-6}
export TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export LOG_DIR=${SAVEPATH}/outputs/StanfordArea5Dataset/slurm/$MODEL_$TIME

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR
LOG="$LOG_DIR/$TIME.txt"

python -m ddp_main \
    --dataset StanfordArea5Dataset \
    --batch_size $BATCH_SIZE \
    --stat_freq 1 --val_freq 100 --save_freq 100 \
    --scheduler PolyLR \
    --model ${MODEL} \
    --conv1_kernel_size 3 \
    --log_dir $LOG_DIR \
    --lr 0.1 \
    --voxel_size 0.05 \
    --max_iter 60000 \
    --data_aug_color_trans_ratio 0.05 \
    --data_aug_color_jitter_std 0.005 \
    --train_phase train \
    --stanford3d_path ${DATAPATH} 2>&1 | tee -a "$LOG"
