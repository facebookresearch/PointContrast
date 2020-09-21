#! /bin/sh
export LR="0.1"
export PRETRAIN="/checkpoint/s9xie/ddp_scannet_logs/FCGF_ScanNet_-default/backup2_02-27/2020-02-24_00-43-38-431206978/checkpoint.pth"
export EXP_NAME="BEST_HARDEST_lr${LR}"
./scripts/launch_cluster_synthia4d.sh

export PRETRAIN="/checkpoint/s9xie/ddp_scannet_logs/FCGF_ScanNet_-default/2020-02-24_00-43-37-223190350/checkpoint.pth"
export EXP_NAME="BEST_NCE_lr${LR}"
./scripts/launch_cluster_synthia4d.sh

export PRETRAIN="/checkpoint/s9xie/ddp_scannet_logs/FCGF_ScanNet_-default/2020-02-27_08-12-00-878101260/checkpoint_23000.pth"
export EXP_NAME="NEW_NCE1_lr${LR}"
./scripts/launch_cluster_synthia4d.sh

export PRETRAIN="/checkpoint/s9xie/ddp_scannet_logs/FCGF_ScanNet_-default/2020-02-27_08-12-00-695572330/checkpoint_22000.pth"
export EXP_NAME="NEW_NCE2_lr${LR}"
./scripts/launch_cluster_synthia4d.sh

export PRETRAIN="/checkpoint/s9xie/ddp_scannet_logs/FCGF_ScanNet_-default/2020-02-27_08-12-03-525856917/checkpoint_23000.pth"
export EXP_NAME="NEW_NCE3_lr${LR}"
./scripts/launch_cluster_synthia4d.sh

export LR="0.8"
export PRETRAIN="/checkpoint/s9xie/ddp_scannet_logs/FCGF_ScanNet_-default/backup2_02-27/2020-02-24_00-43-38-431206978/checkpoint.pth"
export EXP_NAME="BEST_HARDEST_lr${LR}"
./scripts/launch_cluster_synthia4d.sh

export PRETRAIN="/checkpoint/s9xie/ddp_scannet_logs/FCGF_ScanNet_-default/2020-02-24_00-43-37-223190350/checkpoint.pth"
export EXP_NAME="BEST_NCE_lr${LR}"
./scripts/launch_cluster_synthia4d.sh

export PRETRAIN="/checkpoint/s9xie/ddp_scannet_logs/FCGF_ScanNet_-default/2020-02-27_08-12-00-878101260/checkpoint_23000.pth"
export EXP_NAME="NEW_NCE1_lr${LR}"
./scripts/launch_cluster_synthia4d.sh

export PRETRAIN="/checkpoint/s9xie/ddp_scannet_logs/FCGF_ScanNet_-default/2020-02-27_08-12-00-695572330/checkpoint_22000.pth"
export EXP_NAME="NEW_NCE2_lr${LR}"
./scripts/launch_cluster_synthia4d.sh

export PRETRAIN="/checkpoint/s9xie/ddp_scannet_logs/FCGF_ScanNet_-default/2020-02-27_08-12-03-525856917/checkpoint_23000.pth"
export EXP_NAME="NEW_NCE3_lr${LR}"
./scripts/launch_cluster_synthia4d.sh
