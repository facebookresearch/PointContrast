# Copyright (c) Facebook, Inc. and its affiliates.
#  
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash

DATAPATH=~/data/Stanford3D/ # Download Stanford3D dataset and change the path here
PRETRAIN="none"

MODEL=Res16UNet34C
BATCH_SIZE=${BATCH_SIZE:-6}
LOG_DIR=./tmp_dir_stanford

python ddp_main.py \
    train.train_phase=train \
    train.is_train=True \
    train.lenient_weight_loading=True \
    train.stat_freq=1 \
    train.val_freq=200 \
    train.save_freq=100 \
    net.model=${MODEL} \
    net.conv1_kernel_size=3 \
    data.dataset=StanfordArea5Dataset \
    data.batch_size=$BATCH_SIZE \
    data.voxel_size=0.05 \
    data.num_workers=1 \
    data.stanford3d_path=${DATAPATH} \
    augmentation.data_aug_color_trans_ratio=0.05 \
    augmentation.data_aug_color_jitter_std=0.005 \
    optimizer.lr=0.1 \
    optimizer.scheduler=PolyLR \
    optimizer.max_iter=60000 \
    misc.log_dir=${LOG_DIR} \
    distributed=local \
    net.weights=$PRETRAIN \
