"""
Q3 Analysis: Layer Probing — STRUCTURED DATA
=============================================
Same probing as q3_probing.py but for the structured datasets.

Differences from the disordered version:
  - reads from outputs/structured/ .npz files
  - reads structured CSV files (pdb_id, chain_id, annotation)
  - saves to outputs/q3_analysis_structured/

Usage:
  python q3_probing_structured.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
# CONFIG  (structured paths)
# ─────────────────────────────────────────────
NPZ_DIR  = "/work/malekia/esm2-idp-interpretability/outputs/structured"
DATA_DIR = "/work/malekia/esm2-idp-interpretability/data/csv_prepared_data"
OUT_DIR  = "/work/malekia/esm2-idp-interpretability/outputs/q3_analysis_structured"

# Structured train/test files per binding type
BINDING_TYPES = {
    "ion":     ("ahoj_ion_clustered_train.csv",  "ahoj_ion_clustered_test.csv"),
    "protein": ("scannet_sampled_train_clustered.csv",   "scannet_sampled_test_clustered.csv"),
    "dna":     ("biolip_dna_clustered_train.csv", "biolip_dna_clustered_test.csv"),
    "rna":     ("biolip_rna_clustered_train.csv", "biolip_rna_clustered_test.csv"),
}

N_LAYERS = 34
EMB_DIM  = 1280


# ─────────────────────────────────────────────
# DATA BUILDING  (structured CSV format)
# ─────────────────────────────────────────────
def parse_labels(annotation_str):
    """Structured: annotation is a plain string '0110' (no commas)."""
    return np.array([int(c) for c in str(annotation_str)], dtype=np.int8)


def build_layer_dataset(csv_file, layer):
    df = pd.read_csv(os.path.join(DATA_DIR, csv_file))  # CSV
    # protein_id = pdb_id + chain_id
    df['protein_id'] = df['pdb_id'].astype(str) + '_' + df['chain_id'].astype(str)

    X_list, y_list, missing = [], [], 0

    for _, row in df.iterrows():
        protein_id = str(row['protein_id'])
        npz_path = os.path.join(NPZ_DIR, f"{protein_id}.npz")
        if not os.path.exists(npz_path):
            missing += 1
            continue

        data = np.load(npz_path)
        emb = data["embeddings"]           # [34 x L x 1280]
        labels = parse_labels(row['annotation'])

        if emb.shape[1] != len(labels):
            missing += 1
            continue

        X_list.append(emb[layer].astype(np.float32))
        y_list.append(labels)

    if not X_list:
        return None, None, missing

    return np.vstack(X_list), np.concatenate(y_list), missing


# ─────────────────────────────────────────────
# PROBE
# ─────────────────────────────────────────────
def train_and_evaluate(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    probe = LogisticRegression(
        max_iter=3000,
        class_weight='balanced',
        C=1.0,
        n_jobs=-1
    )
    probe.fit(X_train, y_train)

    y_prob = probe.predict_proba(X_test)[:, 1]
    y_pred = probe.predict(X_test)

    return {
        "auroc": roc_auc_score(y_test, y_prob),
        "auprc": average_precision_score(y_test, y_prob),
        "f1":    f1_score(y_test, y_pred, zero_division=0),
        "mcc":   matthews_corrcoef(y_test, y_pred),
    }


# ─────────────────────────────────────────────
# PER BINDING TYPE
# ─────────────────────────────────────────────
def probe_binding_type(binding_type, train_file, test_file):
    print(f"\n{'='*60}")
    print(f"  Binding type: {binding_type.upper()}  (structured)")
    print(f"{'='*60}")

    results = {"layers": [], "auroc": [], "auprc": [], "f1": [], "mcc": []}

    print(f"  {'Layer':<6} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} {'MCC':>8}")
    print(f"  {'-'*42}")

    for layer in range(N_LAYERS):
        X_train, y_train, _ = build_layer_dataset(train_file, layer)
        X_test,  y_test,  _ = build_layer_dataset(test_file, layer)

        if X_train is None or X_test is None:
            print(f"  L{layer:<5} — skipped (no data)")
            continue

        m = train_and_evaluate(X_train, y_train, X_test, y_test)
        results["layers"].append(layer)
        for k in ["auroc", "auprc", "f1", "mcc"]:
            results[k].append(m[k])

        print(f"  L{layer:<5} {m['auroc']:>8.3f} {m['auprc']:>8.3f} "
              f"{m['f1']:>8.3f} {m['mcc']:>8.3f}")

    if results["auprc"]:
        best_idx = int(np.argmax(results["auprc"]))
        best_layer = results["layers"][best_idx]
        print(f"\n  Best layer (AUPRC): {best_layer} "
              f"(AUPRC {results['auprc'][best_idx]:.3f})")
        print(f"  Last layer 33 AUPRC: {results['auprc'][-1]:.3f}")

    return results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}

    for binding_type, (train_file, test_file) in BINDING_TYPES.items():
        save_path = os.path.join(OUT_DIR, f"{binding_type}_q3_results.npz")
        if os.path.exists(save_path):
            print(f"\n  Skipping {binding_type} - already done")
            continue

        results = probe_binding_type(binding_type, train_file, test_file)
        all_results[binding_type] = results

        np.savez(
            save_path,
            layers=np.array(results["layers"]),
            auroc=np.array(results["auroc"]),
            auprc=np.array(results["auprc"]),
            f1=np.array(results["f1"]),
            mcc=np.array(results["mcc"]),
        )
        print(f"  Saved: {save_path}")

    # Summary by AUPRC (primary metric)
    print(f"\n{'='*60}")
    print(f"  CROSS-TYPE SUMMARY (structured, best layer by AUPRC)")
    print(f"{'='*60}")
    print(f"  {'Type':<10} {'Best layer':>12} {'Best AUPRC':>12} {'Layer 33':>12}")
    print(f"  {'-'*48}")
    for btype, res in all_results.items():
        if res["auprc"]:
            best_idx = int(np.argmax(res["auprc"]))
            print(f"  {btype:<10} {res['layers'][best_idx]:>12} "
                  f"{res['auprc'][best_idx]:>12.3f} {res['auprc'][-1]:>12.3f}")

    print(f"\n  Q3 structured analysis complete. Results in {OUT_DIR}")


if __name__ == "__main__":
    main()