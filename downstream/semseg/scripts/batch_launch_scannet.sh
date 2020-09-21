#! /bin/sh
export LR="0.1"
export PRETRAIN="/checkpoint/s9xie/ddp_scannet_logs/FCGF_ScanNet_-default/backup2_02-27/2020-02-24_00-43-38-431206978/checkpoint.pth"
export EXP_NAME="BEST_HARDEST_lr0.1_RERUN"
./scripts/launch_cluster_scannet.sh

export PRETRAIN="/checkpoint/s9xie/ddp_scannet_logs/FCGF_ScanNet_-default/2020-02-24_00-43-37-223190350/checkpoint.pth"
export EXP_NAME="BEST_NCE_lr0.1_RERUN"
./scripts/launch_cluster_scannet.sh

export PRETRAIN="/checkpoint/s9xie/ddp_scannet_logs/FCGF_ScanNet_-default/2020-02-27_08-12-00-878101260/checkpoint_27500.pth"
export EXP_NAME="NEW_NCE1_lr0.1_RERUN"
./scripts/launch_cluster_scannet.sh

export LR="0.8"
export PRETRAIN="/checkpoint/s9xie/ddp_scannet_logs/FCGF_ScanNet_-default/backup2_02-27/2020-02-24_00-43-38-431206978/checkpoint.pth"
export EXP_NAME="BEST_HARDEST_lr0.8_RERUN"
./scripts/launch_cluster_scannet.sh
