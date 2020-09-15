#! /bin/bash
#################################################################################
#     File Name           :     sbatch_train.sh
#     Description         :     Run this script with 'sbatch sbatch_train.sh'
#################################################################################
#SBATCH --output=/checkpoint/s9xie/jobs_output/slurm-%A_%a.out
#SBATCH --error=/checkpoint/s9xie/jobs_output/slurm-%A_%a.err
#SBATCH --partition=priority
#SBATCH --comment="<ECCV deadline>"
#SBATCH --nodes=1
#SBATCH --array=0-0%1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --mem=100g
#SBATCH --cpus-per-task=10
#SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
#SBATCH --open-mode=append
#SBATCH --time=4320

# The ENV below are only used in distributed training with env:// initialization
# export MASTER_ADDR=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:3}
# export MASTER_PORT=29500

trap_handler () {
   echo "Caught signal: " $1
   # SIGTERM must be bypassed
   if [ "$1" = "TERM" ]; then
       echo "bypass sigterm"
   else
     # Submit a new job to the queue
     echo "Requeuing " $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID
     # SLURM_JOB_ID is a unique representation of the job, equivalent
     # to above
     scontrol requeue $SLURM_JOB_ID
   fi
}


# Install signal handler
trap 'trap_handler USR1' USR1
trap 'trap_handler TERM' TERM

SCRIPT=${1}
bash ${SCRIPT}
