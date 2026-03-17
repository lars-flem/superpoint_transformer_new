#!/bin/bash
#SBATCH --job-name=spt_dales
#SBATCH --time=16:00:00
#SBATCH --partition=GPUQ        
#SBATCH --gres=gpu:a100:1       
#SBATCH --mem=96G                 
#SBATCH --cpus-per-task=48 

#SBATCH --output=/cluster/home/larshfle/superpoint_transformer_new/logs/train/slurm_%x_%j.out
#SBATCH --error=/cluster/home/larshfle/superpoint_transformer_new/logs/train/slurm_%x_%j.err

#SBATCH --mail-user=larshfle@stud.ntnu.no
#SBATCH --mail-type=ALL

# --- CONFIGURATION ---

module purge

source ~/.bashrc
conda activate spt

cd /cluster/home/larshfle/superpoint_transformer_new

rm -rf /cluster/home/larshfle/datasets/dales_ply/dales/processed/

python src/train.py \
  experiment=semantic/dales \
  datamodule=semantic/dales \
  paths.data_dir=/cluster/home/larshfle/datasets/dales_ply \
  trainer.check_val_every_n_epoch=5 \
  callbacks.model_checkpoint.save_last=True \
  callbacks.model_checkpoint.every_n_epochs=5 \
  callbacks.model_checkpoint.monitor=null \
  test=True \
  logger=csv