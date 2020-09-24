#!/bin/bash

export OUT_DIR=/checkpoint/s9xie/ji_exps/scannet_nce_debug

python ddp_train.py \
	net.model=Res16UNet34C \
	net.conv1_kernel_size=3 \
	opt.lr=0.1 \
        opt.max_iter=20000 \
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
	# trainer.trainer=PointNCELossTrainer \
