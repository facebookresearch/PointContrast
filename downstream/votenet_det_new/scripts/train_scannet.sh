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
  net.backbone=sparseconv \
  data.dataset=scannet \
  data.num_workers=8 \
  data.batch_size=32 \
  data.num_points=40000 \
  data.no_height=True \
  optimizer.learning_rate=0.001 \
  data.voxelization=True \
  data.voxel_size=0.025 \
  misc.log_dir=$LOGDIR \
  net.is_train=True \
  net.weights=$MODEL \
