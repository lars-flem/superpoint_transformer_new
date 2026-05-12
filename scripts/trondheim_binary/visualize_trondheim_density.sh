#!/bin/bash
#SBATCH --job-name=viz_trondheim_density
#SBATCH --output=logs/slurm/viz_trondheim_density_%j.out
#SBATCH --error=logs/slurm/viz_trondheim_density_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=GPUQ

cd /cluster/home/larshfle/superpoint_transformer_new

mkdir -p logs/visualization

source ~/.bashrc
conda activate spt

python scripts/visualize_trondheim_binary.py
