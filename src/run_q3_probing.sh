#!/bin/bash
#SBATCH --account=ksibio
#SBATCH --partition=gpu-bio
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --mem=128000
#SBATCH --gpus=1
#SBATCH --job-name="esm2-q3-probing"
#SBATCH --output=/work/malekia/esm2-idp-interpretability/logs/q3_probing_output.txt

source ~/miniconda3/etc/profile.d/conda.sh
conda activate esm2-idp
export TORCH_HOME=/work/malekia/torch_cache

python /work/malekia/esm2-idp-interpretability/src/q3_probing.py
