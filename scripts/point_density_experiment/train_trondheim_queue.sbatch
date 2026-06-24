#!/bin/bash
#SBATCH --job-name=spt_trondheim_queue
#SBATCH --time=10:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=10
#SBATCH --array=0-5%1

#SBATCH --output=/cluster/home/jakobep/superpoint_transformer_new/logs/train/slurm_%x_%A_%a.out
#SBATCH --error=/cluster/home/jakobep/superpoint_transformer_new/logs/train/slurm_%x_%A_%a.err

#SBATCH --mail-user=jakobep@stud.ntnu.no
#SBATCH --mail-type=ALL

module purge

source ~/.bashrc
conda activate sptez

cd /cluster/home/jakobep/superpoint_transformer_new

EXPERIMENTS=(
    "semantic/trondheim20"
    "semantic/trondheim30"
    "semantic/trondheim30down"
    "semantic/trondheim20_multiclass"
    "semantic/trondheim30_multiclass"
    "semantic/trondheim30down_multiclass"
#    "semantic/trondheim10_multiclass"
#    "semantic/trondheim10"
)
DATAMODULES=(
    "semantic/trondheim"
    "semantic/trondheim"
    "semantic/trondheim"
    "semantic/trondheim_multiclass"
    "semantic/trondheim_multiclass"
    "semantic/trondheim_multiclass"
#    "semantic/trondheim_multiclass"
#    "semantic/trondheim"
)
DATA_DIRS=(
    "/cluster/home/jakobep/datasets/trondheim20"
    "/cluster/home/jakobep/datasets/trondheim30"
    "/cluster/home/jakobep/datasets/trondheim30down"
    "/cluster/home/jakobep/datasets/trondheim20_mc"
    "/cluster/home/jakobep/datasets/trondheim30_mc"
    "/cluster/home/jakobep/datasets/trondheim30down_mc"
#    "/cluster/home/jakobep/datasets/trondheim10_mc"
#    "/cluster/home/jakobep/datasets/trondheim10"
)

EXPERIMENT=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}
DATAMODULE=${DATAMODULES[$SLURM_ARRAY_TASK_ID]}
DATA_DIR=${DATA_DIRS[$SLURM_ARRAY_TASK_ID]}

echo "Running task $SLURM_ARRAY_TASK_ID: experiment=$EXPERIMENT data_dir=$DATA_DIR"

python src/train.py \
    experiment=$EXPERIMENT \
    datamodule=$DATAMODULE \
    paths.data_dir=$DATA_DIR \
    trainer.check_val_every_n_epoch=5 \
    callbacks.model_checkpoint.save_last=True \
    callbacks.model_checkpoint.every_n_epochs=5 \
    callbacks.model_checkpoint.monitor=null \
    test=True \
    logger=many_loggers
