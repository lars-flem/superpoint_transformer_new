#!/bin/bash
#SBATCH --job-name=spt_bergen2022
#SBATCH --time=16:00:00
#SBATCH --partition=GPUQ        
#SBATCH --gres=gpu:a100:1       
#SBATCH --mem=80G                 
#SBATCH --cpus-per-task=8 

#SBATCH --output=/cluster/home/larshfle/superpoint_transformer_new/logs/train/slurm_%x_%j.out
#SBATCH --error=/cluster/home/larshfle/superpoint_transformer_new/logs/train/slurm_%x_%j.err

#SBATCH --mail-user=larshfle@stud.ntnu.no
#SBATCH --mail-type=ALL

# --- CONFIGURATION ---

module purge

source ~/.bashrc
conda activate spt

cd /cluster/home/larshfle/superpoint_transformer_new

rm -rf /cluster/home/larshfle/datasets/bergen2022_10pkt/processed/ 

source ~/.bashrc && conda activate spt && cd /cluster/home/larshfle/superpoint_transformer_new && python src/train.py \
    experiment=semantic/norway_bergen2022 \
    datamodule=semantic/norway_bergen2022 \
    paths.data_dir=/cluster/home/larshfle/datasets/bergen2022_10pkt \
    datamodule.dataloader.num_workers=8 \
    trainer.check_val_every_n_epoch=5 \
    callbacks.model_checkpoint.save_last=True \
    callbacks.model_checkpoint.every_n_epochs=5 \
    callbacks.model_checkpoint.monitor=null \
    test=True \
    logger=csv 2>&1 | tee /tmp/train_output.log
    
    
#trainer.max_epochs=20 \
