#!/bin/bash
#SBATCH --account=ksibio
#SBATCH --partition=gpu-bio
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --mem=64000
#SBATCH --gpus=1
#SBATCH --job-name="esm2-timing"
#SBATCH --output=/work/malekia/esm2-idp-interpretability/logs/timing_output.txt

source ~/miniconda3/etc/profile.d/conda.sh
conda activate esm2-idp
export TORCH_HOME=/work/malekia/torch_cache

python /work/malekia/esm2-idp-interpretability/src/time_pipeline.py
