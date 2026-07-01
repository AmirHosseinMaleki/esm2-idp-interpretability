"""
Cluster Merged AHoJ Ion Data and Create Clean Splits
=====================================================
Takes the merged AHoJ file (one row per protein chain) and:
  1. Clusters sequences at 10% identity with mmseqs2
  2. Creates cluster-aware train/val/test splits (no leakage)
  3. Samples ~N proteins for the ESM2 pipeline

This mirrors the approach used for ScanNet/BioLiP, fixing the earlier
issue where duplicate rows survived clustering.

Usage:
  python cluster_ahoj.py
"""

import os
import json
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR    = "/work/malekia/esm2-idp-interpretability/data/csv_prepared_data"
WORK_DIR    = "/work/malekia/esm2-idp-interpretability/mmseqs_ahoj"
MMSEQS      = "/work/malekia/esm2-idp-interpretability/software/software/MMseqs2/mmseqs/bin/mmseqs"

INPUT_FILE  = "ahoj_ion_merged.csv"

MIN_SEQ_ID  = 0.1     # 10% identity threshold (same as other datasets)
COVERAGE    = 0.8     # -c 0.8 (same as ScanNet/BioLiP scripts)
THREADS     = 8

SAMPLE_SIZE = 2000    # how many proteins to keep for the pipeline
RANDOM_SEED = 42


# ─────────────────────────────────────────────
# STEP 1 — Build FASTA from unique sequences
# ─────────────────────────────────────────────
def build_fasta(df, fasta_path):
    """One FASTA entry per (pdb_id, chain_id)."""
    print("Building FASTA file...")
    with open(fasta_path, 'w') as f:
        for _, row in df.iterrows():
            header = f"{row['pdb_id']}_{row['chain_id']}"
            f.write(f">{header}\n{row['sequence']}\n")
    print(f"  Wrote {len(df):,} sequences to {fasta_path}")


# ─────────────────────────────────────────────
# STEP 2 — Run mmseqs2 clustering
# ─────────────────────────────────────────────
def run_clustering(fasta_path):
    """Run mmseqs easy-cluster at 10% identity."""
    print("\nRunning mmseqs2 clustering at 10% identity...")
    print("(this may take 10-30 minutes for 111k sequences)")

    cluster_prefix = os.path.join(WORK_DIR, "clusterRes")
    tmp_dir        = os.path.join(WORK_DIR, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    cmd = [
        MMSEQS, "easy-cluster",
        fasta_path,
        cluster_prefix,
        tmp_dir,
        "--min-seq-id", str(MIN_SEQ_ID),
        "-c", str(COVERAGE),
        "--cov-mode", "0",
        "--threads", str(THREADS),
    ]

    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("  ERROR running mmseqs2:")
        print(result.stderr[-2000:])
        raise RuntimeError("mmseqs2 clustering failed")

    cluster_tsv = cluster_prefix + "_cluster.tsv"
    print(f"  Clustering complete: {cluster_tsv}")
    return cluster_tsv


# ─────────────────────────────────────────────
# STEP 3 — Parse clusters
# ─────────────────────────────────────────────
def parse_clusters(cluster_tsv):
    """Map each protein to its cluster representative."""
    print("\nParsing clusters...")
    protein_to_cluster = {}
    clusters = defaultdict(list)

    with open(cluster_tsv) as f:
        for line in f:
            rep, member = line.strip().split('\t')
            protein_to_cluster[member] = rep
            clusters[rep].append(member)

    print(f"  {len(clusters):,} clusters from {len(protein_to_cluster):,} proteins")
    return protein_to_cluster, clusters


# ─────────────────────────────────────────────
# STEP 4 — Cluster-aware split (one representative per cluster)
# ─────────────────────────────────────────────
def make_splits(df, protein_to_cluster):
    """
    Keep ONE representative per cluster, then split by cluster
    so no two splits share similar sequences.
    """
    print("\nCreating cluster-aware splits...")

    df = df.copy()
    df['protein_id'] = df['pdb_id'] + '_' + df['chain_id']
    df['cluster'] = df['protein_id'].map(protein_to_cluster)

    # Proteins not in any cluster → own cluster
    missing = df['cluster'].isna().sum()
    if missing:
        df.loc[df['cluster'].isna(), 'cluster'] = df.loc[df['cluster'].isna(), 'protein_id']

    # Keep ONE representative row per cluster (removes redundancy)
    df_unique = df.drop_duplicates(subset='cluster', keep='first')
    print(f"  After keeping 1 per cluster: {len(df_unique):,} proteins")

    # Split by cluster (here each cluster = one row, so simple split)
    clusters = df_unique['cluster'].unique()
    train_c, temp_c = train_test_split(clusters, test_size=0.30, random_state=RANDOM_SEED)
    val_c, test_c   = train_test_split(temp_c, test_size=0.50, random_state=RANDOM_SEED)

    train = df_unique[df_unique['cluster'].isin(train_c)]
    val   = df_unique[df_unique['cluster'].isin(val_c)]
    test  = df_unique[df_unique['cluster'].isin(test_c)]

    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
    return train, val, test


# ─────────────────────────────────────────────
# STEP 5 — Sample down to target size (stratified by length)
# ─────────────────────────────────────────────
def sample_proteins(train, val, test, total_target):
    """
    Sample proportionally from each split to reach total_target proteins,
    stratified across length ranges so the sample is representative.
    """
    print(f"\nSampling to ~{total_target} proteins total...")

    # Keep the 70/15/15 ratio in the sample
    n_train = int(total_target * 0.70)
    n_val   = int(total_target * 0.15)
    n_test  = int(total_target * 0.15)

    def stratified_sample(df, n):
        if len(df) <= n:
            return df
        # Bin by length into quartiles, sample evenly across bins
        df = df.copy()
        df['len_bin'] = pd.qcut(df['length'], q=4, labels=False, duplicates='drop')
        per_bin = n // df['len_bin'].nunique()
        sampled = df.groupby('len_bin', group_keys=False).apply(
            lambda g: g.sample(min(len(g), per_bin), random_state=RANDOM_SEED)
        )
        # Top up if rounding left us short
        if len(sampled) < n:
            remaining = df[~df.index.isin(sampled.index)]
            extra = remaining.sample(min(len(remaining), n - len(sampled)),
                                     random_state=RANDOM_SEED)
            sampled = pd.concat([sampled, extra])
        return sampled.drop(columns='len_bin')

    train_s = stratified_sample(train, n_train)
    val_s   = stratified_sample(val, n_val)
    test_s  = stratified_sample(test, n_test)

    print(f"  Sampled — Train: {len(train_s):,}  Val: {len(val_s):,}  Test: {len(test_s):,}")
    return train_s, val_s, test_s


# ─────────────────────────────────────────────
# STEP 6 — Save
# ─────────────────────────────────────────────
def save_splits(train, val, test, prefix):
    """Save train/val/test with a given filename prefix."""
    cols = ['pdb_id', 'chain_id', 'sequence', 'annotation', 'length', 'binding_sites']
    out = {
        f"{prefix}_train.csv": train,
        f"{prefix}_val.csv":   val,
        f"{prefix}_test.csv":  test,
    }
    print(f"\nSaving splits ({prefix})...")
    for fname, df in out.items():
        path = os.path.join(DATA_DIR, fname)
        df[cols].to_csv(path, index=False)
        total_res = df['length'].sum()
        bind = df['binding_sites'].sum()
        print(f"  {fname}: {len(df):,} proteins, "
              f"{bind:,}/{total_res:,} binding ({100*bind/total_res:.2f}%)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    os.makedirs(WORK_DIR, exist_ok=True)

    # Load merged data
    df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE))
    print(f"Loaded {len(df):,} merged AHoJ proteins\n")

    # Cluster (reuse existing output if it exists — clustering is expensive)
    fasta_path = os.path.join(WORK_DIR, "ahoj_ion.fasta")
    cluster_tsv = os.path.join(WORK_DIR, "clusterRes_cluster.tsv")
    if os.path.exists(cluster_tsv):
        print(f"Reusing existing clustering: {cluster_tsv}")
    else:
        build_fasta(df, fasta_path)
        cluster_tsv = run_clustering(fasta_path)
    protein_to_cluster, clusters = parse_clusters(cluster_tsv)

    # Split (full clustered, no sampling yet)
    train, val, test = make_splits(df, protein_to_cluster)

    # Save the FULL clustered version (all clusters, no sampling)
    save_splits(train, val, test, prefix="ahoj_ion_clustered_full")

    # Sample down and save the SAMPLED version
    train_s, val_s, test_s = sample_proteins(train, val, test, SAMPLE_SIZE)
    save_splits(train_s, val_s, test_s, prefix="ahoj_ion_clustered")

    print("\n✓ AHoJ clustering + sampling complete")
    print(f"  Full version:    ahoj_ion_clustered_full_{{train,val,test}}.csv")
    print(f"  Sampled version: ahoj_ion_clustered_{{train,val,test}}.csv")
    print(f"  Location: {DATA_DIR}")


if __name__ == "__main__":
    main()