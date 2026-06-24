#!/bin/bash
#SBATCH --job-name=spt_trondheim_5_10_xt3_queue
#SBATCH --time=10:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --constraint=gpu80g
#SBATCH --mem=80G
#SBATCH --cpus-per-task=10
#SBATCH --array=0-3%1

#SBATCH --output=/cluster/home/jakobep/superpoint_transformer_new/logs/train/slurm_%x_%A_%a.out
#SBATCH --error=/cluster/home/jakobep/superpoint_transformer_new/logs/train/slurm_%x_%A_%a.err

#SBATCH --mail-user=jakobep@stud.ntnu.no
#SBATCH --mail-type=ALL

# Queue: trondheim 5pkt and 10pkt × {binary, multiclass} with xy_tiling=3.
# The 20/30/30down runs at xy_tiling=3 are already done; this picks up the
# remaining densities. xy_tiling is overridden on the command line because the
# trondheim{5,10}*.yaml experiment configs still have xy_tiling=2 hardcoded.

module purge

source ~/.bashrc
conda activate sptez

# Fall back to the explicit env interpreter if `conda activate` silently no-oped
# (has happened in non-interactive sbatch shells, manifesting as
# `ModuleNotFoundError: No module named 'pycut_pursuit'`).
PY=/cluster/home/jakobep/.conda/envs/sptez/bin/python

cd /cluster/home/jakobep/superpoint_transformer_new

EXPERIMENTS=(
    "semantic/trondheim5"
    "semantic/trondheim10"
    "semantic/trondheim5_multiclass"
    "semantic/trondheim10_multiclass"
)
DATAMODULES=(
    "semantic/trondheim"
    "semantic/trondheim"
    "semantic/trondheim_multiclass"
    "semantic/trondheim_multiclass"
)
DATA_DIRS=(
    "/cluster/home/jakobep/datasets/trondheim5"
    "/cluster/home/jakobep/datasets/trondheim10"
    "/cluster/home/jakobep/datasets/trondheim5_mc"
    "/cluster/home/jakobep/datasets/trondheim10_mc"
)

EXPERIMENT=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}
DATAMODULE=${DATAMODULES[$SLURM_ARRAY_TASK_ID]}
DATA_DIR=${DATA_DIRS[$SLURM_ARRAY_TASK_ID]}

echo "Running task $SLURM_ARRAY_TASK_ID: experiment=$EXPERIMENT data_dir=$DATA_DIR (xy_tiling=3)"

$PY src/train.py \
    experiment=$EXPERIMENT \
    datamodule=$DATAMODULE \
    datamodule.xy_tiling=3 \
    paths.data_dir=$DATA_DIR \
    trainer.check_val_every_n_epoch=5 \
    callbacks.model_checkpoint.save_last=True \
    callbacks.model_checkpoint.every_n_epochs=5 \
    callbacks.model_checkpoint.monitor=null \
    test=True \
    logger=many_loggers
