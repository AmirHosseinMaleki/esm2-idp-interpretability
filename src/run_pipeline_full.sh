#!/bin/bash
#SBATCH --account=ksibio
#SBATCH --partition=gpu-bio
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=128000
#SBATCH --gpus=1
#SBATCH --job-name="esm2-pipeline-full"
#SBATCH --output=/work/malekia/esm2-idp-interpretability/logs/pipeline_full_output.txt

source ~/miniconda3/etc/profile.d/conda.sh
conda activate esm2-idp
export TORCH_HOME=/work/malekia/torch_cache

BASE=/work/malekia/esm2-idp-interpretability
DATA=$BASE/data/csv_prepared_data
OUT=$BASE/outputs/esm2-idp-interpretability

echo "Started: $(date)"

# Run all 12 TSV files in one job
# Model loads once, processes all files sequentially
python $BASE/src/pipeline.py \
    --output $OUT \
    --max_length 3000 \
    --input \
        $DATA/ion_binding_train.tsv \
        $DATA/ion_binding_val.tsv \
        $DATA/ion_binding_test.tsv \
        $DATA/protein_binding_train.tsv \
        $DATA/protein_binding_val.tsv \
        $DATA/protein_binding_test.tsv \
        $DATA/dna_binding_train.tsv \
        $DATA/dna_binding_val.tsv \
        $DATA/dna_binding_test.tsv \
        $DATA/rna_binding_train.tsv \
        $DATA/rna_binding_val.tsv \
        $DATA/rna_binding_test.tsv

echo "Finished: $(date)"