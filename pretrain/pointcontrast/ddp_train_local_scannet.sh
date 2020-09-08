#!/bin/bash

export LOG_DIR=outputs

python3 ddp_train.py \
	--model=Res16UNet34C \
        --conv1_kernel_size=3 \
	--trainer=PointNCELossTrainer \
	--trainer=HardestContrastiveLossTrainer \
	--lr=0.1 \
	--dataset=ScanNetHardMatchPairDataset \
	--subset_length=0 \
	--batch_size=32 \
	--voxel_size=0.025 \
	--test_valid=False \
	--hit_ratio_thresh=0.04 \
	--free_rot=True \
	--use_color_feat=False \
	--val_batch_size=8 \
	--num_gpus=8 \
        --npos=4096 \
        --nneg=2048 \
	--no_additional_neg=True \
        --use_all_positives=False \
        --use_all_negatives=False \
        --self_contrast=True \
        --nceT=0.4 \
        --stat_freq=1 \
        --lr_update_freq=250 \
        --max_iter=100000 \
	--val_iter_freq=2000 \
        --out_dir $LOG_DIR \
	--use_color_feat=True \
	--infinite_sampler \
