#!/bin/bash
#SBATCH --job-name=viz_trondheim_density
#SBATCH --output=/cluster/home/jakobep/superpoint_transformer_new/logs/visualization/slurm_%x_%j.out
#SBATCH --error=/cluster/home/jakobep/superpoint_transformer_new/logs/visualization/slurm_%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --partition=GPUQ

cd /cluster/home/jakobep/superpoint_transformer_new

mkdir -p visualizations/trondheim_density

source ~/.bashrc
conda activate sptez

python scripts/point_density_experiment/visualize_trondheim_density.py
