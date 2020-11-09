# Copyright (c) Facebook, Inc. and its affiliates.
#  
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash

DATAPATH=~/data/scannet # Download ScanNet segmentation dataset and change the path here
PRETRAIN="none" # For finetuning, use the checkpoint path here.
MODEL=Res16UNet34C
BATCH_SIZE=${BATCH_SIZE:-6}
LOG_DIR=./tmp_dir_scannet

python ddp_main.py \
    train.train_phase=train \
    train.is_train=True \
    train.lenient_weight_loading=True \
    train.stat_freq=1 \
    train.val_freq=500 \
    train.save_freq=500 \
    net.model=${MODEL} \
    net.conv1_kernel_size=3 \
    augmentation.normalize_color=True \
    data.dataset=ScannetVoxelization2cmDataset \
    data.batch_size=$BATCH_SIZE \
    data.num_workers=1 \
    data.num_val_workers=1 \
    data.scannet_path=${DATAPATH} \
    data.return_transformation=False \
    test.test_original_pointcloud=False \
    test.save_prediction=False \
    optimizer.lr=0.8 \
    optimizer.scheduler=PolyLR \
    optimizer.max_iter=60000 \
    misc.log_dir=${LOG_DIR} \
    distributed=local \
    net.weights=${PRETRAIN} \

