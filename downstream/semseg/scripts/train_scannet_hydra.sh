#!/bin/bash

DATAPATH=${3:-"/checkpoint/jihou/data/scannet/pointcloud/"}
PRETRAIN="/private/home/jihou/Mink16UNet34C_ScanNet.pth"
MODEL=Res16UNet34C
BATCH_SIZE=${BATCH_SIZE:-6}
LOG_DIR="/checkpoint/jihou/scannet_baseline"

python ddp_main.py -m \
    train.train_phase=train \
    train.is_train=True \
    train.lenient_weight_loading=True \
    train.stat_freq=1 \
    train.val_freq=200 \
    train.save_freq=100 \
    net.model=${MODEL} \
    net.conv1_kernel_size=3 \
    data.dataset=ScannetVoxelization2cmDataset \
    data.batch_size=$BATCH_SIZE \
    data.num_workers=1 \
    data.scannet_path=${DATAPATH} \
    data.return_transformation=False \
    test.test_original_pointcloud=False \
    test.save_prediction=False \
    optimizer.lr=0.1,0.8 \
    optimizer.scheduler=PolyLR \
    optimizer.max_iter=20000 \
    misc.log_dir=${LOG_DIR} \
    distributed=slurm \
    #hydra.launcher.partition=learnfair \
    #net.weights=${PRETRAIN} \

