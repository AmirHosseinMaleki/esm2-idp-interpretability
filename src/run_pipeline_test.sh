#!/bin/bash
#SBATCH --account=ksibio
#SBATCH --partition=gpu-bio
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --mem=32000
#SBATCH --gpus=1
#SBATCH --job-name="esm2-pipeline-test"
#SBATCH --output=/work/malekia/esm2-idp-interpretability/logs/pipeline_test_output.txt

source ~/miniconda3/etc/profile.d/conda.sh
conda activate esm2-idp
export TORCH_HOME=/work/malekia/torch_cache

BASE=/work/malekia/esm2-idp-interpretability
DATA=$BASE/data/csv_prepared_data
OUT=$BASE/outputs/esm2-idp-interpretability

echo "===== TEST 1: Ion binding ====="
python $BASE/src/pipeline.py --input $DATA/ion_binding_train.tsv --output $OUT --idx 0

echo "===== TEST 2: Protein binding ====="
python $BASE/src/pipeline.py --input $DATA/protein_binding_train.tsv --output $OUT --idx 0

echo "===== TEST 3: DNA binding ====="
python $BASE/src/pipeline.py --input $DATA/dna_binding_train.tsv --output $OUT --idx 0

echo "===== TEST 4: RNA binding ====="
python $BASE/src/pipeline.py --input $DATA/rna_binding_train.tsv --output $OUT --idx 0

echo "===== ALL TESTS COMPLETE ====="
