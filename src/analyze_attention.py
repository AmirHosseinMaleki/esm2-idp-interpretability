"""
Q1 Analysis: Attention at Binding Sites (GPU-accelerated)
==========================================================
Metrics:
  1. Enrichment:     how much more attention do binding site residues receive?
  2. Within-site:    do binding site residues attend to each other?

GPU is used for all tensor operations.
I/O (loading .npz) remains on CPU — unavoidable bottleneck.

Usage:
  python q1_attention_analysis.py
"""

import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
NPZ_DIR  = "/work/malekia/esm2-idp-interpretability/outputs/esm2-idp-interpretability"
DATA_DIR = "/work/malekia/esm2-idp-interpretability/data/csv_prepared_data"
OUT_DIR  = "/work/malekia/esm2-idp-interpretability/outputs/q1_analysis"

BINDING_TYPES = {
    "ion":     ["ion_binding_train.tsv",     "ion_binding_val.tsv",     "ion_binding_test.tsv"],
    "protein": ["protein_binding_train.tsv", "protein_binding_val.tsv", "protein_binding_test.tsv"],
    "dna":     ["dna_binding_train.tsv",     "dna_binding_val.tsv",     "dna_binding_test.tsv"],
    "rna":     ["rna_binding_train.tsv",     "rna_binding_val.tsv",     "rna_binding_test.tsv"],
}

N_LAYERS = 33
N_HEADS  = 20


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_tsv_proteins(tsv_files):
    dfs = []
    for f in tsv_files:
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            dfs.append(pd.read_csv(path, sep='\t'))
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset='protein_id')
    return df


def load_attention_gpu(protein_id, device):
    """
    Load attention tensor from .npz and move to GPU.
    Returns None if file missing or embeddings-only.
    """
    path = os.path.join(NPZ_DIR, f"{protein_id}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    if "attentions" not in data:
        return None
    # Load float16 from disk, convert to float32 on GPU
    attn = torch.from_numpy(data["attentions"].astype(np.float32))
    return attn.to(device)  # [33 x 20 x L x L] on GPU


def parse_labels(labels_str, device):
    """Parse label string and return as GPU boolean mask."""
    labels = np.array([int(x) for x in labels_str.split(',')], dtype=np.int8)
    return torch.from_numpy(labels).to(device)


# ─────────────────────────────────────────────
# METRICS (all GPU tensor operations)
# ─────────────────────────────────────────────
def compute_enrichment(attentions, labels):
    """
    Metric 1: Enrichment of attention at binding sites.

    attentions: [33 x 20 x L x L] torch tensor on GPU
    labels:     [L] torch tensor on GPU

    Returns: enrichment [33 x 20] on CPU, or None
    """
    binding_mask = labels == 1
    n_binding = binding_mask.sum().item()
    n_total   = len(labels)

    if n_binding == 0 or n_binding == n_total:
        return None

    # Column sum: total attention each residue receives
    # attentions[l, h, i, j] → sum over i → [33 x 20 x L]
    col_sums = attentions.sum(dim=2)

    # Mean attention at binding sites vs all residues
    binding_col = col_sums[:, :, binding_mask].mean(dim=2)   # [33 x 20]
    all_col     = col_sums.mean(dim=2)                        # [33 x 20]

    enrichment = binding_col / (all_col + 1e-8)               # [33 x 20]
    return enrichment.cpu().numpy()


def compute_within_site(attentions, labels):
    """
    Metric 2: Within-binding-site attention clustering.

    attentions: [33 x 20 x L x L] torch tensor on GPU
    labels:     [L] torch tensor on GPU

    Returns: (within_fraction [33 x 20] on CPU, baseline float), or None
    """
    binding_mask = labels == 1
    n_binding = binding_mask.sum().item()
    n_total   = len(labels)

    if n_binding == 0 or n_binding == n_total:
        return None

    baseline = n_binding / n_total

    # Rows from binding site residues: [33 x 20 x n_binding x L]
    binding_rows = attentions[:, :, binding_mask, :]

    # Fraction going to binding site columns
    to_binding      = binding_rows[:, :, :, binding_mask]     # [33 x 20 x n_b x n_b]
    within_fraction = to_binding.sum(dim=3).mean(dim=2)       # [33 x 20]

    return within_fraction.cpu().numpy(), baseline


# ─────────────────────────────────────────────
# PER BINDING TYPE ANALYSIS
# ─────────────────────────────────────────────
def analyze_binding_type(binding_type, tsv_files, device):
    print(f"\n{'='*60}")
    print(f"  Binding type: {binding_type.upper()}")
    print(f"{'='*60}")

    df = load_tsv_proteins(tsv_files)
    print(f"  Proteins: {len(df)}")

    all_enrichments = []
    all_within      = []
    all_baselines   = []

    skipped_no_npz  = 0
    skipped_no_attn = 0
    skipped_cov     = 0
    processed       = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {binding_type}"):
        protein_id = str(row['protein_id'])

        # Load attention to GPU
        attentions = load_attention_gpu(protein_id, device)
        if attentions is None:
            path = os.path.join(NPZ_DIR, f"{protein_id}.npz")
            if not os.path.exists(path):
                skipped_no_npz += 1
            else:
                skipped_no_attn += 1
            continue

        # Labels to GPU
        labels = parse_labels(row['labels'], device)

        # Metric 1
        enrichment = compute_enrichment(attentions, labels)
        if enrichment is not None:
            all_enrichments.append(enrichment)

        # Metric 2
        result = compute_within_site(attentions, labels)
        if result is not None:
            within_fraction, baseline = result
            all_within.append(within_fraction)
            all_baselines.append(baseline)
        else:
            skipped_cov += 1

        # Free GPU memory for next protein
        del attentions, labels
        torch.cuda.empty_cache()

        processed += 1

    print(f"\n  Processed:            {processed}")
    print(f"  Skipped (no .npz):    {skipped_no_npz}")
    print(f"  Skipped (emb only):   {skipped_no_attn}")
    print(f"  Skipped (100% cov):   {skipped_cov}")
    print(f"  Enrichment computed:  {len(all_enrichments)}")
    print(f"  Within-site computed: {len(all_within)}")

    results = {}

    if all_enrichments:
        enrichments_arr = np.stack(all_enrichments)        # [n x 33 x 20]
        results["mean_enrichment"]   = enrichments_arr.mean(axis=0)
        results["median_enrichment"] = np.median(enrichments_arr, axis=0)
        results["std_enrichment"]    = enrichments_arr.std(axis=0)
        results["n_enrichment"]      = np.array([len(all_enrichments)])

        layer_max  = results["mean_enrichment"].max(axis=1)
        best_layer = layer_max.argmax() + 1
        best_head  = results["mean_enrichment"][layer_max.argmax()].argmax()
        print(f"\n  [Metric 1] Best layer: {best_layer}  "
              f"Best head: {best_head}  "
              f"Enrichment: {layer_max.max():.3f}x")

        print(f"  Layer summary (max across heads):")
        print(f"  ", end="")
        for l in [0, 4, 8, 12, 16, 20, 24, 28, 32]:
            print(f"  L{l+1:<3}", end="")
        print()
        print(f"  ", end="")
        for l in [0, 4, 8, 12, 16, 20, 24, 28, 32]:
            print(f"  {layer_max[l]:.3f}", end="")
        print()

    if all_within:
        within_arr = np.stack(all_within)
        baselines  = np.array(all_baselines)
        results["mean_within"]   = within_arr.mean(axis=0)
        results["median_within"] = np.median(within_arr, axis=0)
        results["std_within"]    = within_arr.std(axis=0)
        results["mean_baseline"] = np.array([baselines.mean()])
        results["n_within"]      = np.array([len(all_within)])

        layer_max_w  = results["mean_within"].max(axis=1)
        best_layer_w = layer_max_w.argmax() + 1
        best_val_w   = layer_max_w.max()
        baseline_avg = baselines.mean()
        print(f"\n  [Metric 2] Best layer: {best_layer_w}  "
              f"Within-site: {best_val_w:.3f}  "
              f"({best_val_w/baseline_avg:.1f}x over random {baseline_avg:.3f})")

    return results


# ─────────────────────────────────────────────
# CROSS-TYPE SUMMARY
# ─────────────────────────────────────────────
def print_summary(all_results):
    print(f"\n{'='*60}")
    print(f"  CROSS-TYPE SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Type':<10} {'Best layer':>10} {'Enrichment':>12} "
          f"{'Within-site':>13} {'vs random':>10}")
    print(f"  {'-'*55}")

    for btype, res in all_results.items():
        if not res:
            continue
        best_layer = best_enrich = best_within = vs_random = "N/A"

        if "mean_enrichment" in res:
            lmax       = res["mean_enrichment"].max(axis=1)
            best_layer = str(lmax.argmax() + 1)
            best_enrich = f"{lmax.max():.3f}x"

        if "mean_within" in res:
            lmax_w     = res["mean_within"].max(axis=1)
            best_within = f"{lmax_w.max():.3f}"
            baseline   = res["mean_baseline"][0]
            vs_random  = f"{lmax_w.max()/baseline:.1f}x"

        print(f"  {btype:<10} {best_layer:>10} {best_enrich:>12} "
              f"{best_within:>13} {vs_random:>10}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}

    for binding_type, tsv_files in BINDING_TYPES.items():
        results = analyze_binding_type(binding_type, tsv_files, device)
        all_results[binding_type] = results

        if results:
            save_path = os.path.join(OUT_DIR, f"{binding_type}_q1_results.npz")
            np.savez(save_path, **{k: v for k, v in results.items()
                                   if isinstance(v, np.ndarray)})
            print(f"  Saved: {save_path}")

    print_summary(all_results)
    print(f"\n✓ Q1 analysis complete")


if __name__ == "__main__":
    main()