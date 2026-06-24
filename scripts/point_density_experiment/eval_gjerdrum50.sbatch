#!/bin/bash
#SBATCH --job-name=spt_gjerdrum50_eval
#SBATCH --time=04:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --constraint=gpu80g
#SBATCH --mem=80G
#SBATCH --cpus-per-task=10

#SBATCH --output=/cluster/home/jakobep/superpoint_transformer_new/logs/eval/slurm_%x_%j.out
#SBATCH --error=/cluster/home/jakobep/superpoint_transformer_new/logs/eval/slurm_%x_%j.err

#SBATCH --mail-user=jakobep@stud.ntnu.no
#SBATCH --mail-type=ALL

module purge

source ~/.bashrc
conda activate sptez

cd /cluster/home/jakobep/superpoint_transformer_new

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs/eval

python src/eval.py \
    experiment=semantic/gjerdrum50 \
    datamodule=semantic/gjerdrum \
    paths.data_dir=/cluster/home/jakobep/datasets/gjerdrum50 \
    logger=csv \
    ckpt_path=/cluster/home/jakobep/superpoint_transformer_new/logs/train/runs/2026-04-28_13-41-26/checkpoints/last.ckpt
