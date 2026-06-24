#!/bin/bash
#SBATCH --job-name=spt_bergen2022_steep_gm_queue
#SBATCH --time=04:00:00
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

# Re-run bergen2022_steep_3class twice: once with ground_model=ransac, once
# with ground_model=knn. The two settings produce different pre_transform
# cache hashes, so each task preprocesses into its own hash dir — they do
# not clobber each other.
#
# Task 0 also wipes the existing processed/ once at the start to clear out
# any stale .h5 files from the earlier (incorrect) class mapping that had
# road → ground instead of road → not_ground.

module purge

source ~/.bashrc
conda activate sptez

PY=/cluster/home/jakobep/.conda/envs/sptez/bin/python

cd /cluster/home/jakobep/superpoint_transformer_new

DATA_DIR=/cluster/home/jakobep/datasets/bergen2022_steep_3class

GROUND_MODELS=("ransac" "knn")
GM=${GROUND_MODELS[$SLURM_ARRAY_TASK_ID]}

if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
    echo "Task 0: wiping processed/ to clear stale .h5 from old class mapping"
    rm -rf $DATA_DIR/processed/
fi

echo "Running task $SLURM_ARRAY_TASK_ID: ground_model=$GM"

$PY src/train.py \
    experiment=semantic/bergen2022_steep_3class \
    datamodule=semantic/bergen2022_steep_3class \
    paths.data_dir=$DATA_DIR \
    datamodule.ground_model=$GM \
    trainer.check_val_every_n_epoch=5 \
    callbacks.model_checkpoint.save_last=True \
    callbacks.model_checkpoint.every_n_epochs=5 \
    callbacks.model_checkpoint.monitor=null \
    test=True \
    logger=many_loggers
