#!/bin/bash
JOB_NAME="SSL2_scannet_ddp"

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
LOG_DIR="/checkpoint/s9xie/3d_ssl2_scannet/${EXP_NAME}"
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
       --export=ALL,PRETRAIN=${PRETRAIN},LOADFILTER=${LOADFILTER},LR=${LR} \
       --comment="ECCV deadline" \
       /private/home/s9xie/jiatao/SpatioTemporalSegmentation/scripts/train_scannet_ddp_ssl.sh 

echo "Finished launching job ..."



