"""
Q1 Analysis: Attention at Binding Sites — STRUCTURED DATA
=========================================================
Same analysis as q1_attention_analysis.py but for the structured
protein datasets (AHoJ ion, ScanNet protein, BioLiP DNA/RNA).

Differences from the disordered version:
  - reads from outputs/structured/ .npz files
  - reads structured CSV files (pdb_id, chain_id, annotation)
  - saves to outputs/q1_analysis_structured/

Usage:
  python q1_attention_structured.py
"""

import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# ─────────────────────────────────────────────
# CONFIG  (structured paths)
# ─────────────────────────────────────────────
NPZ_DIR  = "/work/malekia/esm2-idp-interpretability/outputs/structured"
DATA_DIR = "/work/malekia/esm2-idp-interpretability/data/csv_prepared_data"
OUT_DIR  = "/work/malekia/esm2-idp-interpretability/outputs/q1_analysis_structured"

# Structured files per binding type (CSV format: pdb_id, chain_id, annotation)
BINDING_TYPES = {
    "ion":     ["ahoj_ion_clustered_train.csv",  "ahoj_ion_clustered_val.csv",  "ahoj_ion_clustered_test.csv"],
    "protein": ["scannet_sampled_train_clustered.csv",   "scannet_sampled_val_clustered.csv",   "scannet_sampled_test_clustered.csv"],
    "dna":     ["biolip_dna_clustered_train.csv", "biolip_dna_clustered_val.csv", "biolip_dna_clustered_test.csv"],
    "rna":     ["biolip_rna_clustered_train.csv", "biolip_rna_clustered_val.csv", "biolip_rna_clustered_test.csv"],
}

N_LAYERS = 33
N_HEADS  = 20


# ─────────────────────────────────────────────
# DATA LOADING  (structured CSV format)
# ─────────────────────────────────────────────
def load_structured_proteins(csv_files):
    """Load proteins from structured CSV files, add a protein_id column."""
    dfs = []
    for f in csv_files:
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))   # CSV → comma separated
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    # Build protein_id from pdb_id + chain_id (matches .npz filenames)
    df['protein_id'] = df['pdb_id'].astype(str) + '_' + df['chain_id'].astype(str)
    df = df.drop_duplicates(subset='protein_id')
    return df


def load_attention_gpu(protein_id, device):
    path = os.path.join(NPZ_DIR, f"{protein_id}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    if "attentions" not in data:
        return None
    attn = torch.from_numpy(data["attentions"].astype(np.float32))
    return attn.to(device)


def parse_labels(annotation_str, device):
    """Structured format: annotation is a plain string '0110' (no commas)."""
    labels = np.array([int(c) for c in str(annotation_str)], dtype=np.int8)
    return torch.from_numpy(labels).to(device)


# ─────────────────────────────────────────────
# METRICS (identical to disordered version)
# ─────────────────────────────────────────────
def compute_enrichment(attentions, labels):
    binding_mask = labels == 1
    n_binding = binding_mask.sum().item()
    n_total   = len(labels)
    if n_binding == 0 or n_binding == n_total:
        return None
    col_sums    = attentions.sum(dim=2)
    binding_col = col_sums[:, :, binding_mask].mean(dim=2)
    all_col     = col_sums.mean(dim=2)
    enrichment  = binding_col / (all_col + 1e-8)
    return enrichment.cpu().numpy()


def compute_within_site(attentions, labels):
    binding_mask = labels == 1
    n_binding = binding_mask.sum().item()
    n_total   = len(labels)
    if n_binding == 0 or n_binding == n_total:
        return None
    baseline = n_binding / n_total
    binding_rows    = attentions[:, :, binding_mask, :]
    to_binding      = binding_rows[:, :, :, binding_mask]
    within_fraction = to_binding.sum(dim=3).mean(dim=2)
    return within_fraction.cpu().numpy(), baseline


# ─────────────────────────────────────────────
# PER BINDING TYPE
# ─────────────────────────────────────────────
def analyze_binding_type(binding_type, csv_files, device):
    print(f"\n{'='*60}")
    print(f"  Binding type: {binding_type.upper()}  (structured)")
    print(f"{'='*60}")

    df = load_structured_proteins(csv_files)
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

        attentions = load_attention_gpu(protein_id, device)
        if attentions is None:
            path = os.path.join(NPZ_DIR, f"{protein_id}.npz")
            if not os.path.exists(path):
                skipped_no_npz += 1
            else:
                skipped_no_attn += 1
            continue

        labels = parse_labels(row['annotation'], device)

        # Guard: label length must match attention L
        if len(labels) != attentions.shape[-1]:
            del attentions
            torch.cuda.empty_cache()
            continue

        enrichment = compute_enrichment(attentions, labels)
        if enrichment is not None:
            all_enrichments.append(enrichment)

        result = compute_within_site(attentions, labels)
        if result is not None:
            within_fraction, baseline = result
            all_within.append(within_fraction)
            all_baselines.append(baseline)
        else:
            skipped_cov += 1

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
        arr = np.stack(all_enrichments)
        results["mean_enrichment"]   = arr.mean(axis=0)
        results["median_enrichment"] = np.median(arr, axis=0)
        results["std_enrichment"]    = arr.std(axis=0)
        results["n_enrichment"]      = np.array([len(all_enrichments)])

        layer_max  = results["mean_enrichment"].max(axis=1)
        best_layer = layer_max.argmax() + 1
        best_head  = results["mean_enrichment"][layer_max.argmax()].argmax()
        print(f"\n  [Metric 1] Best layer: {best_layer}  Best head: {best_head}  "
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
        arr = np.stack(all_within)
        baselines = np.array(all_baselines)
        results["mean_within"]   = arr.mean(axis=0)
        results["median_within"] = np.median(arr, axis=0)
        results["std_within"]    = arr.std(axis=0)
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


def print_summary(all_results):
    print(f"\n{'='*60}")
    print(f"  CROSS-TYPE SUMMARY (structured)")
    print(f"{'='*60}")
    print(f"  {'Type':<10} {'Best layer':>10} {'Enrichment':>12} "
          f"{'Within-site':>13} {'vs random':>10}")
    print(f"  {'-'*55}")
    for btype, res in all_results.items():
        if not res:
            continue
        best_layer = best_enrich = best_within = vs_random = "N/A"
        if "mean_enrichment" in res:
            lmax = res["mean_enrichment"].max(axis=1)
            best_layer = str(lmax.argmax() + 1)
            best_enrich = f"{lmax.max():.3f}x"
        if "mean_within" in res:
            lmax_w = res["mean_within"].max(axis=1)
            best_within = f"{lmax_w.max():.3f}"
            vs_random = f"{lmax_w.max()/res['mean_baseline'][0]:.1f}x"
        print(f"  {btype:<10} {best_layer:>10} {best_enrich:>12} "
              f"{best_within:>13} {vs_random:>10}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}

    for binding_type, csv_files in BINDING_TYPES.items():
        save_path = os.path.join(OUT_DIR, f"{binding_type}_q1_results.npz")
        if os.path.exists(save_path):
            print(f"\n  Skipping {binding_type} - already done")
            continue
        results = analyze_binding_type(binding_type, csv_files, device)
        all_results[binding_type] = results
        if results:
            np.savez(save_path, **{k: v for k, v in results.items()
                                   if isinstance(v, np.ndarray)})
            print(f"  Saved: {save_path}")

    print_summary(all_results)
    print(f"\n  Q1 structured analysis complete")


if __name__ == "__main__":
    main()