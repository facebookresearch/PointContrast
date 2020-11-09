# Copyright (c) Facebook, Inc. and its affiliates.
#  
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash

export OUT_DIR=./tmp_out_dir

python ddp_train.py \
	net.model=Res16UNet34C \
	net.conv1_kernel_size=3 \
	opt.lr=0.1 \
        opt.max_iter=60000 \
	data.dataset=ScanNetMatchPairDataset \
	data.voxel_size=0.025 \
	trainer.batch_size=32 \
        trainer.stat_freq=1 \
        trainer.lr_update_freq=250 \
	misc.num_gpus=8 \
        misc.npos=4096 \
        misc.nceT=0.4 \
	misc.out_dir=${OUT_DIR} \
	trainer.trainer=HardestContrastiveLossTrainer \
        data.dataset_root_dir=~/pointcontrast/pretrain/pointcontrast/example_dataset \
        data.scannet_match_dir=overlap-30-50p-subset.txt \
	# trainer.trainer=PointNCELossTrainer \
