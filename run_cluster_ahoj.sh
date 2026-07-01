#!/bin/bash
#SBATCH --account=ksibio
#SBATCH --partition=gpu-bio
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --job-name="cluster-ahoj"
#SBATCH --output=/work/malekia/esm2-idp-interpretability/logs/cluster_ahoj_output.txt

source ~/miniconda3/etc/profile.d/conda.sh
conda activate esm2-idp

python /work/malekia/esm2-idp-interpretability/src/cluster_ahoj.py
