#!/bin/bash
#SBATCH --job-name=spt_trondheim30down_queue
#SBATCH --time=10:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --gres-flags=enforce-binding
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

# Fall back to the explicit env interpreter if `conda activate` silently no-oped
# (has happened in non-interactive sbatch shells, manifesting as
# `ModuleNotFoundError: No module named 'pycut_pursuit'`).
PY=/cluster/home/jakobep/.conda/envs/sptez/bin/python

cd /cluster/home/jakobep/superpoint_transformer_new

EXPERIMENTS=(
    "semantic/trondheim30down"
    "semantic/trondheim30down_multiclass"
)
DATAMODULES=(
    "semantic/trondheim"
    "semantic/trondheim_multiclass"
)
DATA_DIRS=(
    "/cluster/home/jakobep/datasets/trondheim30down"
    "/cluster/home/jakobep/datasets/trondheim30down_mc"
)

EXPERIMENT=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}
DATAMODULE=${DATAMODULES[$SLURM_ARRAY_TASK_ID]}
DATA_DIR=${DATA_DIRS[$SLURM_ARRAY_TASK_ID]}

echo "Running task $SLURM_ARRAY_TASK_ID: experiment=$EXPERIMENT data_dir=$DATA_DIR"

$PY src/train.py \
    experiment=$EXPERIMENT \
    datamodule=$DATAMODULE \
    paths.data_dir=$DATA_DIR \
    trainer.check_val_every_n_epoch=5 \
    callbacks.model_checkpoint.save_last=True \
    callbacks.model_checkpoint.every_n_epochs=5 \
    callbacks.model_checkpoint.monitor=null \
    test=True \
    logger=many_loggers
