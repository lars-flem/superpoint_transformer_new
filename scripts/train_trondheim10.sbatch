#!/bin/bash
#SBATCH --job-name=spt_trondheim10
#SBATCH --time=72:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=10

#SBATCH --output=/cluster/home/jakobep/superpoint_transformer_new/logs/train/slurm_%x_%j.out
#SBATCH --error=/cluster/home/jakobep/superpoint_transformer_new/logs/train/slurm_%x_%j.err

#SBATCH --mail-user=jakobep@stud.ntnu.no
#SBATCH --mail-type=ALL

module purge

source ~/.bashrc
conda activate sptez

cd /cluster/home/jakobep/superpoint_transformer_new

# Uncomment to force re-preprocessing from raw files.
# rm -rf /cluster/home/jakobep/datasets/trondheim10/processed/

python src/train.py \
    experiment=semantic/trondheim10 \
    datamodule=semantic/trondheim \
    paths.data_dir=/cluster/home/jakobep/datasets/trondheim10 \
    trainer.check_val_every_n_epoch=5 \
    callbacks.model_checkpoint.save_last=True \
    callbacks.model_checkpoint.every_n_epochs=5 \
    callbacks.model_checkpoint.monitor=null \
    test=True \
    logger=csv
