#!/bin/bash

SAVEPATH=/checkpoint/jihou/space/debug
DATAPATH=${3:-"/checkpoint/jihou/data/stanford3d/pointcloud"}
PRETRAIN="/checkpoint/s9xie/ji_exps/scannet_nce_20200909/6/weights/checkpoint_20000.pth"
MODEL=Res16UNet34C
BATCH_SIZE=${BATCH_SIZE:-6}
LOG_DIR="/checkpoint/jihou/test_interactive_session"


python ddp_main.py \
    train.train_phase=train \
    train.lenient_weight_loading=True \
    train.stat_freq=1 \
    train.val_freq=200 \
    train.save_freq=100 \
    net.model=${MODEL} \
    net.conv1_kernel_size=3 \
    net.weights=${PRETRAIN} \
    data.dataset=StanfordArea5Dataset \
    data.batch_size=$BATCH_SIZE \
    data.voxel_size=0.05 \
    data.stanford3d_path=${DATAPATH} \
    augmentation.data_aug_color_trans_ratio=0.05 \
    augmentation.data_aug_color_jitter_std=0.005 \
    optimizer.lr=0.1 \
    optimizer.scheduler=PolyLR \
    optimizer.max_iter=60000 \
    misc.log_dir=${LOG_DIR} \
    distributed.distributed_port=-1
