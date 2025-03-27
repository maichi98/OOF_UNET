#!/bin/bash

# Ensure that the script exits if a command fails
set -e


# Activate the conda environment
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate env_oof_unet

# Define the project directory:
DIR_PROJECT="/home/maichi/work/my_projects/LySAIRI/OOF_UNET"

# Run train.py :
python "$DIR_PROJECT/oof_unet/train.py" --fold 0 --num_epochs 2 --n_units 6
