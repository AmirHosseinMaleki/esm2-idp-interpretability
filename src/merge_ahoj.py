"""
Merge AHoJ Ion-Binding Rows by (pdb_id, chain_id)
==================================================
AHoJ stores one row per ligand-binding EVENT, so a single protein chain
appears in multiple rows (one per bound ion). This script merges those
rows into one row per (pdb_id, chain_id), combining their binding
annotations with OR — a residue is a binding site if it binds an ion
in ANY of that chain's structures.

This makes AHoJ structurally identical to the ScanNet/BioLiP datasets
(one row per protein chain, all binding sites combined).

Input:  ahoj_train_data.csv, ahoj_val_data.csv, ahoj_test_data.csv
        (columns: pdb_id, chain_id, ligand, sequence, annotation, length, binding_sites)

Output: ahoj_ion_merged.csv
        (columns: pdb_id, chain_id, sequence, annotation, length, binding_sites)
        — note: 'ligand' column dropped since rows now combine multiple ligands

Usage:
  python merge_ahoj.py
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = "/work/malekia/esm2-idp-interpretability/data/csv_prepared_data"

INPUT_FILES = [
    "ahoj_train_data.csv",
    "ahoj_val_data.csv",
    "ahoj_test_data.csv",
]

OUTPUT_FILE = "ahoj_ion_merged.csv"


def merge_annotations(annotations):
    """
    Combine multiple annotation strings with OR.
    All annotations must be the same length (same sequence).

    Example:
      "00110"  +  "01000"  →  "01110"
    """
    # Convert each annotation string to an integer array
    arrays = [np.array([int(c) for c in a], dtype=np.int8) for a in annotations]

    # OR them all together
    merged = arrays[0].copy()
    for a in arrays[1:]:
        merged = merged | a

    return ''.join(map(str, merged))


def main():
    # ── Load and combine all splits ──
    print("Loading AHoJ files...")
    dfs = []
    for f in INPUT_FILES:
        path = os.path.join(DATA_DIR, f)
        df = pd.read_csv(path)
        print(f"  {f}: {len(df):,} rows")
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows before merge: {len(df_all):,}")
    print(f"Unique (pdb_id, chain_id): {df_all.groupby(['pdb_id','chain_id']).ngroups:,}")

    # ── Sanity check: all rows for a given (pdb, chain) must have same sequence ──
    print("\nVerifying sequence consistency within each (pdb_id, chain_id)...")
    seq_check = df_all.groupby(['pdb_id', 'chain_id'])['sequence'].nunique()
    inconsistent = seq_check[seq_check > 1]
    if len(inconsistent) > 0:
        print(f"  WARNING: {len(inconsistent)} chains have inconsistent sequences!")
        print(f"  These will be skipped to avoid annotation misalignment.")
    else:
        print(f"  OK — all chains have consistent sequences.")

    # ── Merge by (pdb_id, chain_id) ──
    print("\nMerging annotations per (pdb_id, chain_id)...")

    merged_rows = []
    skipped = 0

    for (pdb_id, chain_id), group in df_all.groupby(['pdb_id', 'chain_id']):
        # All rows in this group should have the same sequence
        sequences = group['sequence'].unique()
        if len(sequences) > 1:
            skipped += 1
            continue   # skip inconsistent chains

        sequence = sequences[0]

        # Merge all annotations with OR
        merged_annotation = merge_annotations(group['annotation'].tolist())

        merged_rows.append({
            'pdb_id':        pdb_id,
            'chain_id':      chain_id,
            'sequence':      sequence,
            'annotation':    merged_annotation,
            'length':        len(sequence),
            'binding_sites': merged_annotation.count('1'),
        })

    df_merged = pd.DataFrame(merged_rows)

    print(f"\nRows after merge: {len(df_merged):,}")
    print(f"Skipped (inconsistent sequence): {skipped}")

    # ── Statistics ──
    total_res = df_merged['length'].sum()
    total_bind = df_merged['binding_sites'].sum()
    print(f"\nMerged dataset statistics:")
    print(f"  Proteins:        {len(df_merged):,}")
    print(f"  Total residues:  {total_res:,}")
    print(f"  Binding sites:   {total_bind:,} ({100*total_bind/total_res:.2f}%)")

    # Compare before/after for a sanity check
    print(f"\nBefore merge: avg {df_all['binding_sites'].mean():.1f} binding sites/row")
    print(f"After merge:  avg {df_merged['binding_sites'].mean():.1f} binding sites/row")
    print(f"(After should be higher — multiple ion events combined per protein)")

    # ── Save ──
    out_path = os.path.join(DATA_DIR, OUTPUT_FILE)
    df_merged.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"\nColumn format: {list(df_merged.columns)}")
    print("(matches ScanNet/BioLiP structure — one row per protein chain)")


if __name__ == "__main__":
    main()