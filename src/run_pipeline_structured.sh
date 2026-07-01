#!/bin/bash
#SBATCH --account=ksibio
#SBATCH --partition=gpu-bio
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --mem=128000
#SBATCH --gpus=1
#SBATCH --job-name="esm2-structured"
#SBATCH --output=/work/malekia/esm2-idp-interpretability/logs/pipeline_structured_output.txt

source ~/miniconda3/etc/profile.d/conda.sh
conda activate esm2-idp
export TORCH_HOME=/work/malekia/torch_cache

BASE=/work/malekia/esm2-idp-interpretability
DATA=$BASE/data/csv_prepared_data
OUT=$BASE/outputs/structured

echo "Started: $(date)"

# Process all structured binding types into a SEPARATE output directory
# so they don't mix with the DisProt .npz files
python $BASE/src/pipeline.py \
    --output $OUT \
    --max_length 3000 \
    --input \
        $DATA/ahoj_ion_clustered_train.csv \
        $DATA/ahoj_ion_clustered_val.csv \
        $DATA/ahoj_ion_clustered_test.csv \
        $DATA/biolip_dna_clustered_train.csv \
        $DATA/biolip_dna_clustered_val.csv \
        $DATA/biolip_dna_clustered_test.csv \
        $DATA/biolip_rna_clustered_train.csv \
        $DATA/biolip_rna_clustered_val.csv \
        $DATA/biolip_rna_clustered_test.csv \
        $DATA/scannet_train_clustered.csv \
        $DATA/scannet_val_clustered.csv \
        $DATA/scannet_test_clustered.csv

echo "Finished: $(date)"