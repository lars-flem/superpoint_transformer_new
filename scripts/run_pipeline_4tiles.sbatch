#!/bin/bash
#SBATCH --job-name=national_pipeline
#SBATCH --time=0:20:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

#SBATCH --output=/cluster/home/larshfle/superpoint_transformer_new/logs/train/slurm_%x_%j.out
#SBATCH --error=/cluster/home/larshfle/superpoint_transformer_new/logs/train/slurm_%x_%j.err

#SBATCH --mail-user=larshfle@stud.ntnu.no
#SBATCH --mail-type=ALL

module purge
source ~/.bashrc
cd /cluster/home/larshfle/superpoint_transformer_new

PYTHON=/cluster/home/larshfle/.conda/envs/spt/bin/python

$PYTHON scripts/run_national_pipeline.py

echo "Alle tiles ferdig!"
