#!/bin/bash

set -x
# Exit script when a command returns nonzero state
set -e

set -o pipefail

SAVEPATH=/checkpoint/s9xie/space/3d_ssl2
DATAPATH=${3:-"/private/home/s9xie/data/3d_ssl2/Stanford3D/"}
# PRETRAIN="/checkpoint/s9xie/fcgf_logs/FCGF_-default/2019-11-18_18-14-50/checkpoint.pth"
# PRETRAIN="/checkpoint/s9xie/pointcontrast_scannet_logs/POINTCONTRAST_SCN_-default/2020-01-29_17-03-45/checkpoint.pth"
# PRETRAIN="/checkpoint/s9xie/pretrained_weights/Mink16UNet34C_ScanNet.pth"
# PRETRAIN="/private/home/s9xie/3d_ssl2/outputs/checkpoint.pth"
# PRETRAIN="/checkpoint/s9xie/SN_notrans_FCGF_0078_bs32_lr0.010_pm1.500_vs0.02500_npts2048/checkpoint.pth"
# PRETRAIN="/checkpoint/s9xie/pointcontrast_scannet_logs/POINTCONTRAST_SCN_-default/2020-02-10_20-05-19/checkpoint_13.pth"
# PRETRAIN="/checkpoint/s9xie/pointcontrast_scannet_logs/POINTCONTRAST_SCN_-default/2020-02-10_20-04-28/checkpoint_14.pth"
# PRETRAIN="/checkpoint/s9xie/pointcontrast_scannet_logs/POINTCONTRAST_SCN_-default/2020-02-11_00-28-12/checkpoint_33.pth"
# PRETRAIN="/checkpoint/s9xie/ddp_pointcontrast_scannet_logs/POINTCONTRAST_SCN_-default/2020-02-12_21-10-39/checkpoint.pth"
# LOADFILTER="all_bn"

MODEL=Res16UNet34C
# MODEL=ResUNetBN2C

export PYTHONUNBUFFERED="True"
# export CUDA_VISIBLE_DEVICES=$1
export BATCH_SIZE=${BATCH_SIZE:-6}
export TIME=$(date +"%Y-%m-%d_%H-%M-%S-%N")
export LOG_DIR=${SAVEPATH}/outputs/StanfordArea5Dataset/slurm/SSL_${MODEL}_$TIME

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR
LOG="$LOG_DIR/$TIME.txt"


python -m ddp_main \
    --dataset StanfordArea5Dataset \
    --batch_size $BATCH_SIZE \
    --stat_freq 1 --val_freq 200 --save_freq 100 \
    --scheduler PolyLR \
    --model ${MODEL} \
    --conv1_kernel_size 3 \
    --log_dir $LOG_DIR \
    --lr 0.1 \
    --voxel_size 0.05 \
    --load_bn ${LOADFILTER} \
    --max_iter 60000 \
    --data_aug_color_trans_ratio 0.05 \
    --data_aug_color_jitter_std 0.005 \
    --train_phase train \
    --weights ${PRETRAIN} --lenient_weight_loading "True" \
    --stanford3d_path ${DATAPATH} 2>&1 | tee -a "$LOG"
