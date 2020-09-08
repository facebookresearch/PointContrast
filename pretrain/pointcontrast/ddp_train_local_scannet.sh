#!/bin/bash

export LOG_DIR=./outputs

python3 ddp_train.py \
    logging.out_dir=$LOG_DIR \
	net.model=Res16UNet34C \
    net.conv1_kernel_size=3 \
	opt.lr=0.1 \
    opt.max_iter=100000 \
	data.dataset=ScanNetHardMatchPairDataset \
	data.voxel_size=0.025 \
	trainer.trainer=HardestContrastiveLossTrainer \
	trainer.subset_length=0 \
	trainer.batch_size=32 \
	trainer.test_valid=False \
	trainer.hit_ratio_thresh=0.04 \
	trainer.val_batch_size=8 \
    trainer.stat_freq=1 \
    trainer.lr_update_freq=250 \
	trainer.val_iter_freq=2000 \
	trainer.infinite_sampler=True\
	misc.free_rot=True \
	misc.num_gpus=8 \
    misc.npos=4096 \
    misc.nneg=2048 \
	misc.no_additional_neg=True \
    misc.use_all_positives=False \
    misc.use_all_negatives=False \
    misc.self_contrast=True \
    misc.nceT=0.4 \
	misc.use_color_feat=True \
