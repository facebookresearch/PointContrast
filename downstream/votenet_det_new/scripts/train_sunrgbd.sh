# Copyright (c) Facebook, Inc. and its affiliates.
#  
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


#! /bin/bash
export MODEL=<MODEL_DIR_WITH_PRETRAINED_CHECKPOINT>
export LOGDIR=<YOUR_LOG_DIR>
mkdir -p $LOGDIR

# main script
python ddp_main.py \
  net.is_train=True \
  net.backbone=sparseconv \
  data.dataset=sunrgbd \
  data.num_workers=8 \
  data.batch_size=64 \
  data.no_height=True \
  data.voxelization=True \
  data.voxel_size=0.025 \
  optimizer.learning_rate=0.001 \
  misc.log_dir=$LOGDIR \
  net.weights=$MODEL \
