#!/bin/bash

export OUT_DIR=/checkpoint/s9xie/ji_exps/scannet_nce_20200909

python ddp_train.py -m \
    net.model=Res16UNet34C \
    net.conv1_kernel_size=3 \
    opt.lr=0.1 \
    opt.max_iter=20000 \
    data.dataset=ScanNetHardMatchPairDataset,ScanNetMatchPairDataset \
    data.voxel_size=0.025 \
    trainer.trainer=PointNCELossTrainer,HardestContrastiveLossTrainer \
    trainer.batch_size=32 \
    trainer.stat_freq=1 \
    trainer.lr_update_freq=250,1000 \
    misc.num_gpus=8 \
    misc.free_rot=True \
    misc.npos=4096 \
    misc.nceT=0.4 \
    misc.use_color_feat=True \
    misc.out_dir=${OUT_DIR} \
    hydra.launcher.partition=learnfair \
    hydra.launcher.timeout_min=1200 \
    hydra.launcher.max_num_timeout=3 \
    hydra.launcher.signal_delay_s=300 \
    #trainer.trainer=HardestContrastiveLossTrainer \
    #misc.train_num_thread=0 \
    #misc.weight=/checkpoint/jihou/checkpoints/saining_hardest/checkpoint.pth \
    #misc.config=/checkpoint/jihou/2020-09-08/outputs_hardest/outputs_hardest/config.yaml \
