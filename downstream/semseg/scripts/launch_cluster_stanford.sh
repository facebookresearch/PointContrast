#!/bin/bash
JOB_NAME="SSL2_stanford_ddp"
# EXP_NAME="SSL2_stanford_ddp_PCNMatchHardestDDP_color_lr0.1_100ep"
# EXP_NAME="SSL2_stanford_ddp_3DMatchDDP_lr0.8_81ep"
# EXP_NAME="SSL2_stanford_scannet_randomcrop_163ep_randomaug"
# EXP_NAME="SSL2_stanford_ddp_nce_p9182_selfcontrast_noaddneg_t0.07_20ep"

# PCN_Match
# PRETRAIN="/checkpoint/s9xie/pointcontrast_scannet_logs/POINTCONTRAST_SCN_-default/2020-02-11_16-13-45/best_val_checkpoint_20.pth"
# PRETRAIN="/checkpoint/s9xie/pointcontrast_scannet_logs/POINTCONTRAST_SCN_-default/2020-02-11_16-13-45/best_val_checkpoint_40.pth"

# PCN_reg
# PRETRAIN="/checkpoint/s9xie/pointcontrast_scannet_logs/POINTCONTRAST_SCN_-default/2020-02-11_16-54-01/best_val_checkpoint_20.pth"
# PRETRAIN="/checkpoint/s9xie/pointcontrast_scannet_logs/POINTCONTRAST_SCN_-default/2020-02-11_16-54-01/checkpoint_40.pth"

# PCN_free
# PRETRAIN="/checkpoint/s9xie/pointcontrast_scannet_logs/POINTCONTRAST_SCN_-default/2020-02-11_16-53-50/checkpoint_20.pth"
# PRETRAIN="/checkpoint/s9xie/pointcontrast_scannet_logs/POINTCONTRAST_SCN_-default/2020-02-11_16-53-50/checkpoint_40.pth"

# HARDEST PCN Match
# no color
# PRETRAIN="/checkpoint/s9xie/pointcontrast_scannet_logs/POINTCONTRAST_SCN_-default/2020-02-12_14-28-18/checkpoint_15.pth"
# color
# PRETRAIN="/checkpoint/s9xie/pointcontrast_scannet_logs/POINTCONTRAST_SCN_-default/2020-02-12_14-29-51/checkpoint_15.pth"

# HARDEST PCN MATCH DDP
# 0.1
# PRETRAIN="/checkpoint/s9xie/ddp_pointcontrast_scannet_logs/POINTCONTRAST_SCN_-default/2020-02-12_21-10-39/checkpoint.pth"
# 0.2
# PRETRAIN="/checkpoint/s9xie/ddp_pointcontrast_scannet_logs/POINTCONTRAST_SCN_-default/2020-02-12_22-44-09/checkpoint.pth"
# 0.4
# PRETRAIN="/checkpoint/s9xie/ddp_pointcontrast_scannet_logs/POINTCONTRAST_SCN_-default/2020-02-12_22-42-10/checkpoint.pth"
# 0.8
# PRETRAIN="/checkpoint/s9xie/ddp_pointcontrast_scannet_logs/POINTCONTRAST_SCN_-default/2020-02-12_22-37-09/checkpoint.pth"
# color 0.1
# PRETRAIN="/checkpoint/s9xie/ddp_pointcontrast_scannet_logs/POINTCONTRAST_SCN_-default/2020-02-12_20-45-46/checkpoint.pth"

# SINGLE GPU PCN Match 48ep
# PRETRAIN="/checkpoint/s9xie/pointcontrast_scannet_logs/POINTCONTRAST_SCN_-default/2020-02-12_14-28-18/checkpoint_49.pth"

# 3D Match DDP early 19ep
# PRETRAIN="/checkpoint/s9xie/ddp_3dmatch_logs/FCGF_3DMatch_-default/2020-02-13_00-45-11/checkpoint.pth"
# PRETRAIN="/checkpoint/s9xie/ddp_3dmatch_logs/FCGF_4DMatch_-default/2020-02-13_00-45-11/checkpoint.pth"

# FCGF 77 epoch
# PRETRAIN="/checkpoint/s9xie/fcgf_logs/FCGF_-default/2019-11-18_18-14-50/checkpoint.pth"

# Randomcrop
# PRETRAIN="/checkpoint/s9xie/ddp_pointcontrast_scannet_logs/POINTCONTRAST_SCN_-default/2020-02-14_11-49-12/checkpoint.pth"

# Random aug (contrast + global)
# PRETRAIN="/checkpoint/s9xie/ddp_pointcontrast_scannet_logs/POINTCONTRAST_SCN_-default/2020-02-15_23-53-29/checkpoint.pth"


#NCE variants
# p9182_selfcontrast_noaddneg_t0.07
# PRETRAIN="/checkpoint/s9xie/ddp_3dmatch_logs/FCGF_3DMatch_-default/2020-02-20_23-35-24/best_val_checkpoint.pth"

LOADFILTER="all_bn"



# basic configurations
NUM_GPU=8
NUM_JOB=1
WALL_TIME=72
TASK_PER_NODE=1
QUEUE=priority
CPU_PER_GPU=6
NICE=0
NUM_CPU=$((${NUM_GPU} * ${CPU_PER_GPU}))
LOG_DIR="/checkpoint/s9xie/3d_ssl2_stanford/${EXP_NAME}"
USER=s9xie

mkdir -p ${LOG_DIR}

# print job summary
echo "JOB ${JOB_NAME} | EXP ${EXP_NAME}"
echo "LOG DIR ${LOG_DIR}"

# launch
echo "Launch job ..."
sbatch --nodes=1 \
       --cpus-per-task=${NUM_CPU} \
       --mem=512GB \
       --array=1-${NUM_JOB} \
       --ntasks-per-node=${TASK_PER_NODE} \
       --gres=gpu:${NUM_GPU} \
       --time=${WALL_TIME}:00:00 \
       --partition=${QUEUE} \
       --output="${LOG_DIR}/cluster_log_j%A_%a_%N.out" \
       --error="${LOG_DIR}/cluster_log_j%A_%a_%N.err" \
       --job-name=${JOB_NAME} \
       --nice=${NICE} \
       --signal=B:USR1@600 \
       --mail-user=${USER}@fb.com \
       --mail-type=BEGIN,END,FAIL,REQUEUE \
       --constraint=volta \
       --export=ALL,PRETRAIN=${PRETRAIN},LOADFILTER=${LOADFILTER} \
       --comment="ICLR deadline" \
       /private/home/s9xie/PointContrast_release/downstream/S3DIS/scripts/train_stanford_ddp_ssl.sh
#       /private/home/s9xie/jiatao/SpatioTemporalSegmentation/scripts/train_stanford_ddp_ssl.sh 

echo "Finished launching job ..."



