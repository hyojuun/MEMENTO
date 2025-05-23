#!/bin/bash

set -e

ENV_NAME="habitat-datasets"
PYTHON_VERSION="3.9"

echo "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Installing required Python packages with pip..."
pip install --upgrade pip
pip install gitpython tqdm requests

echo "Checking required system packages..."

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 not found. Please install it manually."
        exit 1
    else
        echo "Found $1"
    fi
}

check_command git
check_command git-lfs
check_command wget
check_command curl

echo "Initializing git-lfs..."
git lfs install

echo "Running datasets_download.py for main datasets..."
python scripts/datasets_download.py \
    --uids rearrange_task_assets hab_spot_arm hab3-episodes habitat_humanoids \
    --data-path data/ --no-replace --no-prune

echo "Cloning OVMM objects..."
git clone https://huggingface.co/datasets/ai-habitat/OVMM_objects data/objects_ovmm --recursive

echo "Setting up HSSD scene dataset..."
git clone -b partnr https://huggingface.co/datasets/hssd/hssd-hab data/versioned_data/hssd-hab
cd data/versioned_data/hssd-hab
git lfs pull
cd ../../..
ln -s versioned_data/hssd-hab data/hssd-hab

echo "Downloading and linking partnr_episodes (task dataset and models)..."
git clone https://huggingface.co/datasets/ai-habitat/partnr_episodes data/versioned_data/partnr_episodes
cd data/versioned_data/partnr_episodes
git lfs pull
cd ../../..
mkdir -p data/datasets
ln -s ../versioned_data/partnr_episodes data/datasets/partnr_episodes
ln -s versioned_data/partnr_episodes/checkpoints data/models

echo "Downloading hssd-partnr-ci for testing..."
git clone https://huggingface.co/datasets/ai-habitat/hssd-partnr-ci data/versioned_data/hssd-partnr-ci
ln -s versioned_data/hssd-partnr-ci data/hssd-partnr-ci
cd data/hssd-partnr-ci
git lfs pull
cd ../..

echo "All datasets are successfully downloaded and linked!"
echo "To activate the environment later, run:"
echo "    conda activate $ENV_NAME"