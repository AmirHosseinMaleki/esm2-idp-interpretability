#!/bin/bash
#SBATCH --account=ksibio
#SBATCH --partition=gpu-bio
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --mem=128000
#SBATCH --gpus=1
#SBATCH --job-name="q3-structured"
#SBATCH --output=/work/malekia/esm2-idp-interpretability/logs/q3_structured_output.txt
source ~/miniconda3/etc/profile.d/conda.sh
conda activate esm2-idp
export TORCH_HOME=/work/malekia/torch_cache
python -u /work/malekia/esm2-idp-interpretability/src/q3_probing_structured.py
