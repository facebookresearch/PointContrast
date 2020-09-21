#!/bin/bash

export OUT_DIR=/checkpoint/s9xie/ji_exps/scannet_nce_20200915_lr0.05_infonce_60K_maxiter

python ddp_train.py -m \
    net.model=Res16UNet34C \
    net.conv1_kernel_size=3 \
    opt.lr=0.05 \
    opt.max_iter=60000 \
    data.dataset=ScanNetMatchPairDataset \
    data.voxel_size=0.025 \
    trainer.trainer=PointNCELossTrainer \
    trainer.batch_size=32 \
    trainer.stat_freq=1 \
    trainer.lr_update_freq=250 \
    misc.num_gpus=8 \
    misc.free_rot=True \
    misc.npos=4096 \
    misc.nceT=0.4 \
    misc.use_color_feat=True \
    misc.out_dir=${OUT_DIR} \
    hydra.launcher.partition=dev \
    hydra.launcher.timeout_min=3600 \
    hydra.launcher.max_num_timeout=3 \
    hydra.launcher.signal_delay_s=300 \
    #trainer.trainer=HardestContrastiveLossTrainer \
