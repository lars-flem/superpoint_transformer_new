#!/bin/bash
#SBATCH --job-name=spt_viken2022_eval_cm
#SBATCH --time=00:30:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16

#SBATCH --output=/cluster/home/jakobep/superpoint_transformer_new/logs/eval/slurm_%x_%j.out
#SBATCH --error=/cluster/home/jakobep/superpoint_transformer_new/logs/eval/slurm_%x_%j.err

#SBATCH --mail-user=jakobep@stud.ntnu.no
#SBATCH --mail-type=END,FAIL

module purge

source ~/.bashrc
conda activate sptez

cd /cluster/home/jakobep/superpoint_transformer_new

CKPT=/cluster/home/jakobep/superpoint_transformer_new/logs/train/runs/2026-03-31_23-50-44/checkpoints/last.ckpt

python src/eval.py \
  experiment=semantic/viken2022 \
  datamodule=semantic/viken2022 \
  paths.data_dir=/cluster/home/jakobep/datasets/viken2022 \
  ckpt_path=${CKPT} \
  logger=csv
