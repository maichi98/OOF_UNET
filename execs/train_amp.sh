#!/bin/bash

# Ensure that the script exits if a command fails
set -e


# Activate the conda environment
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate env_oof_unet

# Define the project directory:
DIR_PROJECT="/home/nathan/work/OOF_UNET"

# Run train.py :
python "$DIR_PROJECT/oof_unet/train_baseline_unet_amp.py" --fold 0 --epochs 200 --n_units 6 --num_workers 8 --batches_per_epoch 200
python "$DIR_PROJECT/oof_unet/train_baseline_unet_amp.py" --fold 1 --epochs 200 --n_units 6 --num_workers 8 --batches_per_epoch 200
python "$DIR_PROJECT/oof_unet/train_baseline_unet_amp.py" --fold 2 --epochs 200 --n_units 6 --num_workers 8 --batches_per_epoch 200
python "$DIR_PROJECT/oof_unet/train_baseline_unet_amp.py" --fold 3 --epochs 200 --n_units 6 --num_workers 8 --batches_per_epoch 200
python "$DIR_PROJECT/oof_unet/train_baseline_unet_amp.py" --fold 4 --epochs 200 --n_units 6 --num_workers 8 --batches_per_epoch 200
