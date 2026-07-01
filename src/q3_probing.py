"""
Q3 Analysis: Layer Probing
===========================
Trains a simple logistic regression probe on ESM2 embeddings from
each layer to find WHERE binding site information is best encoded.

Method:
  For each binding type, for each layer (0-33):
    1. Build X_train [N_residues x 1280] from all train proteins
    2. Train logistic regression (class-weighted for imbalance)
    3. Evaluate on test set
    4. Record AUROC, AUPRC, F1, MCC

The hypothesis (from the research project) is that the last layer
performs best. Q3 tests whether that holds, or whether intermediate
layers are already (or more) informative.

Splits use the existing train/test TSV files (<10% sequence similarity).

Usage:
  python q3_probing.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
NPZ_DIR  = "/work/malekia/esm2-idp-interpretability/outputs/esm2-idp-interpretability"
DATA_DIR = "/work/malekia/esm2-idp-interpretability/data/csv_prepared_data"
OUT_DIR  = "/work/malekia/esm2-idp-interpretability/outputs/q3_analysis"

# Train and test files per binding type
# (val not used for probing — could be used for hyperparameter tuning later)
BINDING_TYPES = {
    "ion":     ("ion_binding_train.tsv",     "ion_binding_test.tsv"),
    "protein": ("protein_binding_train.tsv", "protein_binding_test.tsv"),
    "dna":     ("dna_binding_train.tsv",     "dna_binding_test.tsv"),
    "rna":     ("rna_binding_train.tsv",     "rna_binding_test.tsv"),
}

N_LAYERS  = 34   # layers 0 through 33
EMB_DIM   = 1280


# ─────────────────────────────────────────────
# DATA BUILDING
# ─────────────────────────────────────────────
def parse_labels(labels_str):
    return np.array([int(x) for x in labels_str.split(',')], dtype=np.int8)


def build_layer_dataset(tsv_file, layer):
    """
    For a given TSV file and layer, build the stacked residue dataset.

    Returns:
      X [N_residues x 1280]  — embeddings at this layer for all residues
      y [N_residues]         — binding labels

    Loads each protein's .npz, extracts the layer slice, stacks them.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, tsv_file), sep='\t')

    X_list = []
    y_list = []
    missing = 0

    for _, row in df.iterrows():
        protein_id = str(row['protein_id'])
        npz_path = os.path.join(NPZ_DIR, f"{protein_id}.npz")

        if not os.path.exists(npz_path):
            missing += 1
            continue

        data = np.load(npz_path)
        emb = data["embeddings"]          # [34 x L x 1280] float16
        labels = parse_labels(row['labels'])

        # Sanity check — embedding L must match label L
        if emb.shape[1] != len(labels):
            missing += 1
            continue

        # Extract this layer's embeddings: [L x 1280]
        layer_emb = emb[layer].astype(np.float32)

        X_list.append(layer_emb)
        y_list.append(labels)

    if not X_list:
        return None, None, missing

    X = np.vstack(X_list)              # [N_residues x 1280]
    y = np.concatenate(y_list)         # [N_residues]

    return X, y, missing


# ─────────────────────────────────────────────
# PROBE TRAINING + EVALUATION
# ─────────────────────────────────────────────
def train_and_evaluate(X_train, y_train, X_test, y_test):
    """
    Train a class-weighted logistic regression probe and evaluate.
    Returns dict of metrics.
    """
    # Standardize features (important for logistic regression)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Class-weighted logistic regression handles imbalance
    probe = LogisticRegression(
        max_iter=3000,
        class_weight='balanced',   # accounts for rare binding sites
        C=1.0,                      # regularization strength
        n_jobs=-1
    )
    probe.fit(X_train, y_train)

    # Predict probabilities for AUROC/AUPRC
    y_prob = probe.predict_proba(X_test)[:, 1]
    y_pred = probe.predict(X_test)

    metrics = {
        "auroc": roc_auc_score(y_test, y_prob),
        "auprc": average_precision_score(y_test, y_prob),
        "f1":    f1_score(y_test, y_pred, zero_division=0),
        "mcc":   matthews_corrcoef(y_test, y_pred),
    }
    return metrics


# ─────────────────────────────────────────────
# PER BINDING TYPE
# ─────────────────────────────────────────────
def probe_binding_type(binding_type, train_file, test_file):
    print(f"\n{'='*60}")
    print(f"  Binding type: {binding_type.upper()}")
    print(f"{'='*60}")

    # Storage for results across layers
    results = {
        "layers": [],
        "auroc":  [],
        "auprc":  [],
        "f1":     [],
        "mcc":    [],
    }

    print(f"  {'Layer':<6} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} {'MCC':>8}")
    print(f"  {'-'*42}")

    for layer in range(N_LAYERS):
        # Build datasets for this layer
        X_train, y_train, miss_tr = build_layer_dataset(train_file, layer)
        X_test,  y_test,  miss_te = build_layer_dataset(test_file, layer)

        if X_train is None or X_test is None:
            print(f"  L{layer:<5} — skipped (no data)")
            continue

        # Train + evaluate
        metrics = train_and_evaluate(X_train, y_train, X_test, y_test)

        results["layers"].append(layer)
        results["auroc"].append(metrics["auroc"])
        results["auprc"].append(metrics["auprc"])
        results["f1"].append(metrics["f1"])
        results["mcc"].append(metrics["mcc"])

        print(f"  L{layer:<5} {metrics['auroc']:>8.3f} {metrics['auprc']:>8.3f} "
              f"{metrics['f1']:>8.3f} {metrics['mcc']:>8.3f}")

    # Find best layer by AUROC
    if results["auroc"]:
        best_idx = np.argmax(results["auroc"])
        best_layer = results["layers"][best_idx]
        best_auroc = results["auroc"][best_idx]
        last_auroc = results["auroc"][-1]   # layer 33

        print(f"\n  Best layer:  {best_layer} (AUROC {best_auroc:.3f})")
        print(f"  Last layer:  33 (AUROC {last_auroc:.3f})")
        if best_layer == 33:
            print(f"  → Last layer is best (confirms research project assumption)")
        else:
            print(f"  → Layer {best_layer} beats the last layer "
                  f"(+{best_auroc - last_auroc:.3f} AUROC)")

    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}

    for binding_type, (train_file, test_file) in BINDING_TYPES.items():
        results = probe_binding_type(binding_type, train_file, test_file)
        all_results[binding_type] = results

        # Save
        save_path = os.path.join(OUT_DIR, f"{binding_type}_q3_results.npz")
        np.savez(
            save_path,
            layers=np.array(results["layers"]),
            auroc=np.array(results["auroc"]),
            auprc=np.array(results["auprc"]),
            f1=np.array(results["f1"]),
            mcc=np.array(results["mcc"]),
        )
        print(f"  Saved: {save_path}")

    # Cross-type summary
    print(f"\n{'='*60}")
    print(f"  CROSS-TYPE SUMMARY (best layer by AUROC)")
    print(f"{'='*60}")
    print(f"  {'Type':<10} {'Best layer':>12} {'Best AUROC':>12} {'Layer 33 AUROC':>16}")
    print(f"  {'-'*52}")
    for btype, res in all_results.items():
        if res["auroc"]:
            best_idx = np.argmax(res["auroc"])
            print(f"  {btype:<10} {res['layers'][best_idx]:>12} "
                  f"{res['auroc'][best_idx]:>12.3f} {res['auroc'][-1]:>16.3f}")

    print(f"\n Q3 analysis complete. Results in {OUT_DIR}")


if __name__ == "__main__":
    main()