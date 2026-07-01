#!/bin/bash
#SBATCH --account=ksibio
#SBATCH --partition=gpu-bio
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=128000
#SBATCH --gpus=1
#SBATCH --job-name="esm2-q1-analysis"
#SBATCH --output=/work/malekia/esm2-idp-interpretability/logs/q1_analysis_output.txt

source ~/miniconda3/etc/profile.d/conda.sh
conda activate esm2-idp
export TORCH_HOME=/work/malekia/torch_cache

python /work/malekia/esm2-idp-interpretability/src/q1_attention_analysis.py
