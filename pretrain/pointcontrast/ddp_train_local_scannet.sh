#!/bin/bash

export LOG_DIR=outputs

python3 ddp_train.py \
	--model=Res16UNet34C \
        --conv1_kernel_size=3 \
	--trainer=PointNCELossTrainer \
	--trainer=HardestContrastiveLossTrainer \
	--lr=0.1 \
	--dataset=ScanNetHardMatchPairDataset \
	--batch_size=32 \
	--voxel_size=0.025 \
	--free_rot=True \
	--num_gpus=8 \
        --npos=4096 \
        --nceT=0.4 \
        --stat_freq=1 \
        --lr_update_freq=250 \
        --max_iter=100000 \
        --out_dir $LOG_DIR \
