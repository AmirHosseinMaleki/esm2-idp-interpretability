#!/bin/bash
#SBATCH --account=ksibio
#SBATCH --partition=gpu-bio
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --mem=512000
#SBATCH --gpus=1
#SBATCH --job-name="esm2-q2"
#SBATCH --output=/work/malekia/esm2-idp-interpretability/logs/q2_output.txt
source ~/miniconda3/etc/profile.d/conda.sh
conda activate esm2-idp
export TORCH_HOME=/work/malekia/torch_cache
python -u /work/malekia/esm2-idp-interpretability/src/q2_disorder_analysis.py
