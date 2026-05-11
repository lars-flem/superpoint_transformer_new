#!/bin/bash
#SBATCH --job-name=spt_viken2022_eval
#SBATCH --time=10:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=10

#SBATCH --output=/cluster/home/jakobep/superpoint_transformer_new/logs/eval/slurm_%x_%j.out
#SBATCH --error=/cluster/home/jakobep/superpoint_transformer_new/logs/eval/slurm_%x_%j.err

# Optional mail notifications
#SBATCH --mail-user=jakobep@stud.ntnu.no
#SBATCH --mail-type=ALL

module purge

source ~/.bashrc
conda activate sptez

cd /cluster/home/jakobep/superpoint_transformer_new

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs/eval

python src/eval.py \
    experiment=semantic/viken2022 \
    datamodule=semantic/viken2022 \
    paths.data_dir=/cluster/home/jakobep/datasets/viken2022 \
    logger=csv \
    ckpt_path=/cluster/home/jakobep/superpoint_transformer_new/logs/train/runs/2026-03-31_23-50-44/checkpoints/last.ckpt
