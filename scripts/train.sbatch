#!/bin/bash
#SBATCH --account=share-ie-IDI
#SBATCH --job-name=spt_dales
#SBATCH --time=08:00:00
#SBATCH --partition=GPUQ        
#SBATCH --gres=gpu:a100:1       
#SBATCH --mem=80                 
#SBATCH --cpus-per-task=48        
#SBATCH --output=output_label.txt
#SBATCH --error=output_label.err
#SBATCH --mail-user=larshfle@stud.ntnu.no
#SBATCH --mail-type=ALL

# --- CONFIGURATION ---

module purge

source ~/.bashrc
conda activate spt

cd /cluster/home/larshfle/superpoint_transformer_new

python src/train.py \
  experiment=semantic/dales \
  datamodule=semantic/dales \
  datamodule.num_workers=2 \
  paths.data_dir=/cluster/home/larshfle/datasets/dales_ply \
  trainer.check_val_every_n_epoch=5 \
  callbacks.model_checkpoint.save_last=True \
  callbacks.model_checkpoint.every_n_epochs=5 \
  callbacks.model_checkpoint.monitor=null \
  test=True \
  logger=csv