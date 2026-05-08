#!/bin/bash
#SBATCH --job-name=spt_norway_combined_3class_mini
#SBATCH --time=4:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem=40G
#SBATCH --cpus-per-task=10

#SBATCH --output=/cluster/home/jakobep/superpoint_transformer_new/logs/train/slurm_%x_%j.out
#SBATCH --error=/cluster/home/jakobep/superpoint_transformer_new/logs/train/slurm_%x_%j.err

#SBATCH --mail-user=jakobep@stud.ntnu.no
#SBATCH --mail-type=ALL

# Mini run: uses TILES_MINI (5 train / 2 val / 2 test per region = 36 tiles total).
# Intended for quick sanity checks of the pipeline, not for actual model evaluation.

module purge

source ~/.bashrc
conda activate sptez

cd /cluster/home/jakobep/superpoint_transformer_new

# Uncomment to force re-preprocessing from raw files.
# rm -rf /cluster/home/jakobep/datasets/norway_combined_3class/processed/

python src/train.py \
    experiment=semantic/norway_combined_3class \
    datamodule=semantic/norway_combined_3class \
    datamodule.mini=True \
    paths.data_dir=/cluster/home/jakobep/datasets/norway_combined_3class \
    trainer.max_epochs=50 \
    trainer.check_val_every_n_epoch=5 \
    callbacks.model_checkpoint.save_last=True \
    callbacks.model_checkpoint.every_n_epochs=5 \
    callbacks.model_checkpoint.monitor=null \
    test=True \
    logger=many_loggers
