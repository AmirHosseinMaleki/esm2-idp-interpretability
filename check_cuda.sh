#!/bin/bash
#SBATCH --account=ksibio
#SBATCH --partition=gpu-bio
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --mem=4000
#SBATCH --gpus=1
#SBATCH --job-name="check-cuda"
#SBATCH --output=check_cuda_output.txt

nvidia-smi
echo "---"
nvcc --version 2>/dev/null || echo "nvcc not found"
