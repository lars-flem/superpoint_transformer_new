#!/bin/bash
#SBATCH --job-name=downsample_trondheim30down
#SBATCH --time=02:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

#SBATCH --output=/cluster/home/jakobep/superpoint_transformer_new/logs/train/slurm_%x_%j.out
#SBATCH --error=/cluster/home/jakobep/superpoint_transformer_new/logs/train/slurm_%x_%j.err

#SBATCH --mail-user=jepunnerud@gmail.com
#SBATCH --mail-type=END,FAIL

source ~/.bashrc
conda activate sptez

cd /cluster/home/jakobep/superpoint_transformer_new

python scripts/downsample_laz.py \
    --input_dir /cluster/home/jakobep/datasets/trondheim30/raw \
    --output_dir /cluster/home/jakobep/datasets/trondheim30down/raw \
    --density 30 \
    --cell_size 1.0 \
    --seed 42
