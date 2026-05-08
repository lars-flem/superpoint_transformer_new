#!/bin/bash
#SBATCH --job-name=spt_gjerdrum_binary
#SBATCH --time=12:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --constraint=gpu80g
#SBATCH --mem=80G
#SBATCH --cpus-per-task=10
#SBATCH --array=0-1%1

#SBATCH --output=/cluster/home/jakobep/superpoint_transformer_new/logs/train/slurm_%x_%A_%a.out
#SBATCH --error=/cluster/home/jakobep/superpoint_transformer_new/logs/train/slurm_%x_%A_%a.err

#SBATCH --mail-user=jakobep@stud.ntnu.no
#SBATCH --mail-type=ALL

module purge

source ~/.bashrc
conda activate sptez

cd /cluster/home/jakobep/superpoint_transformer_new

EXPERIMENTS=(
    "semantic/gjerdrum5"
    "semantic/gjerdrum50"
)
DATA_DIRS=(
    "/cluster/home/jakobep/datasets/gjerdrum5"
    "/cluster/home/jakobep/datasets/gjerdrum50"
)
WANDB_NAMES=(
    "SPT-64-5pkt"
    "SPT-64-50pkt"
)

EXPERIMENT=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}
DATA_DIR=${DATA_DIRS[$SLURM_ARRAY_TASK_ID]}
WANDB_NAME=${WANDB_NAMES[$SLURM_ARRAY_TASK_ID]}

echo "Running task $SLURM_ARRAY_TASK_ID: experiment=$EXPERIMENT data_dir=$DATA_DIR"

python src/train.py \
    experiment=$EXPERIMENT \
    datamodule=semantic/gjerdrum \
    paths.data_dir=$DATA_DIR \
    trainer.check_val_every_n_epoch=5 \
    callbacks.model_checkpoint.save_last=True \
    callbacks.model_checkpoint.every_n_epochs=5 \
    callbacks.model_checkpoint.monitor=null \
    test=True \
    logger=many_loggers \
    logger.wandb.name=$WANDB_NAME
