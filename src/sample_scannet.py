"""
Sample ScanNet protein-binding data down to ~2000 proteins.
Stratified by length so the sample is representative.
Keeps the train/val/test split structure.
"""
import os
import pandas as pd

DATA_DIR = "/work/malekia/esm2-idp-interpretability/data/csv_prepared_data"
TARGET   = 2000
SEED     = 42

# 70/15/15 split of the target
splits = {
    "train": ("scannet_train_clustered.csv", int(TARGET * 0.70)),
    "val":   ("scannet_val_clustered.csv",   int(TARGET * 0.15)),
    "test":  ("scannet_test_clustered.csv",  int(TARGET * 0.15)),
}

def stratified_sample(df, n):
    if len(df) <= n:
        return df
    df = df.copy()
    df['len_bin'] = pd.qcut(df['length'], q=4, labels=False, duplicates='drop')
    per_bin = n // df['len_bin'].nunique()
    sampled = df.groupby('len_bin', group_keys=False).apply(
        lambda g: g.sample(min(len(g), per_bin), random_state=SEED)
    )
    if len(sampled) < n:
        remaining = df[~df.index.isin(sampled.index)]
        extra = remaining.sample(min(len(remaining), n - len(sampled)), random_state=SEED)
        sampled = pd.concat([sampled, extra])
    return sampled.drop(columns='len_bin')

for split, (fname, n) in splits.items():
    path = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(path)
    sampled = stratified_sample(df, n)
    out_name = fname.replace("scannet_", "scannet_sampled_")
    out_path = os.path.join(DATA_DIR, out_name)
    sampled.to_csv(out_path, index=False)
    print(f"{fname}: {len(df)} -> {len(sampled)}  saved as {out_name}")

print("\nDone. Use scannet_sampled_*_clustered.csv in the analysis scripts.")