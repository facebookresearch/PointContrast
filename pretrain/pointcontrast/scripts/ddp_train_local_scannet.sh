#!/bin/bash

export OUT_DIR=/checkpoint/s9xie/ji_exps/scannet_nce
#export OUT_DIR=/checkpoint/jihou/checkpoints/scannet_nce

python ddp_train.py -m \
	net.model=Res16UNet34C \
    net.conv1_kernel_size=3 \
	opt.lr=0.1 \
    opt.max_iter=20000 \
	data.dataset=ScanNetHardMatchPairDataset,ScanNetMatchPairDataset \
	data.voxel_size=0.025 \
	trainer.trainer=PointNCELossTrainer \
	trainer.subset_length=0 \
	trainer.batch_size=32 \
	trainer.test_valid=False \
	trainer.hit_ratio_thresh=0.1 \
	trainer.val_batch_size=8 \
    trainer.stat_freq=1 \
    trainer.lr_update_freq=250 \
	trainer.val_iter_freq=2000 \
	trainer.infinite_sampler=True\
	misc.num_gpus=8 \
	misc.free_rot=True \
    misc.npos=4096 \
    misc.nneg=2048 \
	misc.no_additional_neg=True \
    misc.use_all_positives=False \
    misc.use_all_negatives=False \
    misc.self_contrast=True \
    misc.nceT=0.4 \
	misc.use_color_feat=True \
	misc.out_dir=${OUT_DIR} \
	#trainer.trainer=HardestContrastiveLossTrainer \
	#misc.train_num_thread=0 \
	#misc.weight=/checkpoint/jihou/checkpoints/saining_hardest/checkpoint.pth \
	#misc.config=/checkpoint/jihou/2020-09-08/outputs_hardest/outputs_hardest/config.yaml \
