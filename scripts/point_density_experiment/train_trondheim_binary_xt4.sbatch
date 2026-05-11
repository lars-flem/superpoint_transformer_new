#!/bin/bash
#SBATCH --job-name=spt_trondheim_binary_xt4
#SBATCH --time=12:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem=120G
#SBATCH --cpus-per-task=10
#SBATCH --array=0-2%1

#SBATCH --output=/cluster/home/jakobep/superpoint_transformer_new/logs/train/slurm_%x_%A_%a.out
#SBATCH --error=/cluster/home/jakobep/superpoint_transformer_new/logs/train/slurm_%x_%A_%a.err

#SBATCH --mail-user=jakobep@stud.ntnu.no
#SBATCH --mail-type=ALL

module purge

source ~/.bashrc
conda activate sptez

cd /cluster/home/jakobep/superpoint_transformer_new

EXPERIMENTS=(
    "semantic/trondheim10"
    "semantic/trondheim20"
    "semantic/trondheim30"
)
DATA_DIRS=(
    "/cluster/home/jakobep/datasets/trondheim10"
    "/cluster/home/jakobep/datasets/trondheim20"
    "/cluster/home/jakobep/datasets/trondheim30"
)
WANDB_NAMES=(
    "SPT-64-10pkt-xt4"
    "SPT-64-20pkt-xt4"
    "SPT-64-30pkt-xt4"
)

EXPERIMENT=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}
DATA_DIR=${DATA_DIRS[$SLURM_ARRAY_TASK_ID]}
WANDB_NAME=${WANDB_NAMES[$SLURM_ARRAY_TASK_ID]}

echo "Running task $SLURM_ARRAY_TASK_ID: experiment=$EXPERIMENT data_dir=$DATA_DIR xy_tiling=4 (enforce-binding)"

python src/train.py \
    experiment=$EXPERIMENT \
    datamodule=semantic/trondheim \
    paths.data_dir=$DATA_DIR \
    trainer.check_val_every_n_epoch=5 \
    callbacks.model_checkpoint.save_last=True \
    callbacks.model_checkpoint.every_n_epochs=5 \
    callbacks.model_checkpoint.monitor=null \
    test=True \
    logger=many_loggers \
    logger.wandb.name=$WANDB_NAME
