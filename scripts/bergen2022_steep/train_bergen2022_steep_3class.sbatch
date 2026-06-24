#!/bin/bash
#SBATCH --job-name=spt_bergen2022_steep_3class
#SBATCH --time=04:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --constraint=gpu80g
#SBATCH --mem=80G
#SBATCH --cpus-per-task=10

#SBATCH --output=/cluster/home/jakobep/superpoint_transformer_new/logs/train/slurm_%x_%j.out
#SBATCH --error=/cluster/home/jakobep/superpoint_transformer_new/logs/train/slurm_%x_%j.err

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

# Uncomment to force re-preprocessing from raw files.
# rm -rf /cluster/home/jakobep/datasets/bergen2022_steep_3class/processed/

$PY src/train.py \
    experiment=semantic/bergen2022_steep_3class \
    datamodule=semantic/bergen2022_steep_3class \
    paths.data_dir=/cluster/home/jakobep/datasets/bergen2022_steep_3class \
    trainer.check_val_every_n_epoch=5 \
    callbacks.model_checkpoint.save_last=True \
    callbacks.model_checkpoint.every_n_epochs=5 \
    callbacks.model_checkpoint.monitor=null \
    test=True \
    logger=many_loggers
