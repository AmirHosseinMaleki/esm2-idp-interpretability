"""
Q2 Analysis: Attention in Disordered vs Structured Regions
===========================================================
Uses DisProt disorder annotations (per-residue: 1=disordered, 0=structured)
together with ESM2 attention to answer three questions:

  Q2.1 - Which layers attend to disordered regions?
         Metric: attention RECEIVED by disordered vs structured residues.
         disorder_preference[layer,head] =
             mean attention received by disordered residues /
             mean attention received by structured residues
         > 1 = head prefers disordered, < 1 = prefers structured.

  Q2.2 - How does attention BEHAVE in disordered vs structured regions?
         Metric: attention ENTROPY (focused vs diffuse).
         Low entropy = focused on few residues; high = spread out.
         Hypothesis: structured = focused (low), disordered = diffuse (high).

  Q2.3 - How does attention behave at the EDGE (boundary)?
         Metric: CROSS-BOUNDARY FLOW as a function of distance from a
         disorder<->structure transition. For each residue at offset d
         from a boundary, what fraction of its attention crosses to the
         other side?

Only "mosaic" proteins (have both disordered AND structured regions) are
used - fully disordered / fully structured proteins have no boundary.

Input:
  - disorder annotations: data/disorder_annotations/<type>_disorder.tsv
  - attention tensors:    outputs/esm2-idp-interpretability/<protein_id>.npz

Output: outputs/q2_analysis/<type>_q2_results.npz

Usage:
  python q2_disorder_analysis.py
"""

import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
NPZ_DIR      = "/work/malekia/esm2-idp-interpretability/outputs/esm2-idp-interpretability"
DISORDER_DIR = "/work/malekia/esm2-idp-interpretability/data/disorder_annotations"
OUT_DIR      = "/work/malekia/esm2-idp-interpretability/outputs/q2_analysis"

BINDING_TYPES = ["ion", "protein", "dna", "rna"]

N_LAYERS = 33
N_HEADS  = 20

# Q2.3 boundary window: analyze residues from -BOUNDARY_WIN to +BOUNDARY_WIN
BOUNDARY_WIN = 20

# Mosaic threshold: keep proteins with disorder coverage in this range
MOSAIC_MIN = 5.0    # %
MOSAIC_MAX = 95.0   # %


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_disorder_proteins(binding_type):
    """Load disorder annotations, keep only mosaic proteins."""
    path = os.path.join(DISORDER_DIR, f"{binding_type}_disorder.tsv")
    df = pd.read_csv(path, sep='\t')
    # Keep mosaic proteins (both regions present)
    mosaic = df[(df['disorder_coverage'] >= MOSAIC_MIN) &
                (df['disorder_coverage'] <= MOSAIC_MAX)].copy()
    return mosaic


def load_attention_gpu(protein_id, device):
    path = os.path.join(NPZ_DIR, f"{protein_id}.npz")
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path)
        if "attentions" not in data:
            return None
        attn = torch.from_numpy(data["attentions"].astype(np.float32))
        return attn.to(device)   # [33 x 20 x L x L]
    except Exception:
        return None


def parse_disorder(disorder_str, device):
    """Disorder labels are comma-separated '0,1,1,0'."""
    arr = np.array([int(x) for x in disorder_str.split(',')], dtype=np.int8)
    return torch.from_numpy(arr).to(device)


# ─────────────────────────────────────────────
# Q2.1 - Disorder preference (attention received)
# ─────────────────────────────────────────────
def compute_disorder_preference(attentions, disorder):
    """
    attention received by disordered vs structured residues.
    Returns [33 x 20] preference, or None.
    """
    dis_mask = disorder == 1
    str_mask = disorder == 0
    if dis_mask.sum() == 0 or str_mask.sum() == 0:
        return None

    # Column sum = attention received by each residue: [33 x 20 x L]
    col_sums = attentions.sum(dim=2)

    dis_attn = col_sums[:, :, dis_mask].mean(dim=2)   # [33 x 20]
    str_attn = col_sums[:, :, str_mask].mean(dim=2)   # [33 x 20]

    preference = dis_attn / (str_attn + 1e-8)
    return preference.cpu().numpy()


# ─────────────────────────────────────────────
# Q2.2 - Attention entropy (disordered vs structured)
# ─────────────────────────────────────────────
def compute_entropy(attentions, disorder):
    """
    Per-residue attention entropy, averaged over disordered vs
    structured residues separately.

    entropy(i) = -sum_j a_ij * log(a_ij)
    Returns (dis_entropy [33x20], str_entropy [33x20]) or None.

    Memory-efficient: computes entropy in-place per layer to avoid
    materializing a full second [33 x 20 x L x L] tensor at once,
    which causes OOM on long proteins.
    """
    dis_mask = disorder == 1
    str_mask = disorder == 0
    if dis_mask.sum() == 0 or str_mask.sum() == 0:
        return None

    L = attentions.shape[-1]
    n_layers = attentions.shape[0]

    # Accumulate entropy [33 x 20 x L] layer by layer to bound memory
    entropy = torch.empty(n_layers, N_HEADS, L, device=attentions.device)
    for layer in range(n_layers):
        a = attentions[layer].clamp(min=1e-9)        # [20 x L x L]
        # -sum_j a*log(a) computed for this layer only
        entropy[layer] = -(a * a.log()).sum(dim=2)   # [20 x L]
        del a

    dis_entropy = entropy[:, :, dis_mask].mean(dim=2)   # [33 x 20]
    str_entropy = entropy[:, :, str_mask].mean(dim=2)   # [33 x 20]

    del entropy
    return dis_entropy.cpu().numpy(), str_entropy.cpu().numpy()


# ─────────────────────────────────────────────
# Q2.3 - Cross-boundary flow vs distance
# ─────────────────────────────────────────────
def find_boundaries(disorder_np):
    """
    Find indices where disorder state changes (transition points).
    Returns list of boundary positions (index of the first residue
    AFTER the transition).
    """
    changes = np.where(np.diff(disorder_np) != 0)[0] + 1
    return changes.tolist()


def compute_cross_boundary(attentions, disorder, win):
    """
    For each residue at offset d (-win..+win) from each boundary,
    compute the fraction of its attention that crosses to the OTHER
    side of that boundary.

    Returns:
      flow_sum  [33 x 20 x (2*win+1)]  summed cross-boundary fractions
      flow_cnt  [(2*win+1)]            count of contributing residues
    or None if no boundaries.
    """
    disorder_np = disorder.cpu().numpy()
    boundaries = find_boundaries(disorder_np)
    if not boundaries:
        return None

    L = len(disorder_np)
    offsets = np.arange(-win, win + 1)
    n_off = len(offsets)

    flow_sum = torch.zeros(N_LAYERS, N_HEADS, n_off, device=attentions.device)
    flow_cnt = np.zeros(n_off, dtype=np.int64)

    for b in boundaries:
        # Side of the boundary: residues < b are one side, >= b the other
        # "Other side" for residue i depends on which side i is on.
        for oi, d in enumerate(offsets):
            i = b + d
            if i < 0 or i >= L:
                continue

            # Which side is residue i on? (relative to THIS boundary b)
            i_side_left = i < b   # True = left side, False = right side

            # Mask of residues on the OTHER side of boundary b
            if i_side_left:
                other_mask = torch.arange(L, device=attentions.device) >= b
            else:
                other_mask = torch.arange(L, device=attentions.device) < b

            # Attention from i to the other side: [33 x 20]
            # attentions[:, :, i, :] is row i → sums to 1
            row = attentions[:, :, i, :]                  # [33 x 20 x L]
            cross = row[:, :, other_mask].sum(dim=2)      # [33 x 20]

            flow_sum[:, :, oi] += cross
            flow_cnt[oi] += 1

    return flow_sum.cpu().numpy(), flow_cnt, offsets


# ─────────────────────────────────────────────
# PER BINDING TYPE
# ─────────────────────────────────────────────
def analyze_binding_type(binding_type, device):
    print(f"\n{'='*60}")
    print(f"  {binding_type.upper()}")
    print(f"{'='*60}")

    df = load_disorder_proteins(binding_type)
    print(f"  Mosaic proteins (5-95% disorder): {len(df)}")

    # Accumulators
    pref_list   = []                      # Q2.1
    dis_ent_list, str_ent_list = [], []   # Q2.2
    flow_sum_total = np.zeros((N_LAYERS, N_HEADS, 2 * BOUNDARY_WIN + 1))  # Q2.3
    flow_cnt_total = np.zeros(2 * BOUNDARY_WIN + 1, dtype=np.int64)
    offsets_global = None

    processed = 0
    skipped   = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {binding_type}"):
        protein_id = str(row['protein_id'])

        attentions = load_attention_gpu(protein_id, device)
        if attentions is None:
            skipped += 1
            continue

        disorder = parse_disorder(row['disorder_labels'], device)

        # Guard: disorder length must match attention L
        if len(disorder) != attentions.shape[-1]:
            del attentions
            torch.cuda.empty_cache()
            skipped += 1
            continue

        # Q2.1
        pref = compute_disorder_preference(attentions, disorder)
        if pref is not None:
            pref_list.append(pref)

        # Q2.2
        ent = compute_entropy(attentions, disorder)
        if ent is not None:
            dis_ent_list.append(ent[0])
            str_ent_list.append(ent[1])

        # Q2.3
        cb = compute_cross_boundary(attentions, disorder, BOUNDARY_WIN)
        if cb is not None:
            flow_sum, flow_cnt, offsets = cb
            flow_sum_total += flow_sum
            flow_cnt_total += flow_cnt
            offsets_global = offsets

        del attentions, disorder
        torch.cuda.empty_cache()
        processed += 1

    print(f"\n  Processed: {processed}   Skipped: {skipped}")

    results = {}

    # ── Q2.1 aggregate ──
    if pref_list:
        pref_arr = np.stack(pref_list)            # [n x 33 x 20]
        results["disorder_preference"] = pref_arr.mean(axis=0)
        lp = results["disorder_preference"].max(axis=1)
        best_layer = lp.argmax() + 1
        print(f"\n  [Q2.1] Disorder preference - best layer {best_layer}, "
              f"max {lp.max():.3f}x")
        print(f"         (>1 = attends more to disordered residues)")

    # ── Q2.2 aggregate ──
    if dis_ent_list:
        dis_ent = np.stack(dis_ent_list).mean(axis=0)   # [33 x 20]
        str_ent = np.stack(str_ent_list).mean(axis=0)   # [33 x 20]
        results["disordered_entropy"] = dis_ent
        results["structured_entropy"] = str_ent
        # Average across heads for a summary
        dis_mean = dis_ent.mean()
        str_mean = str_ent.mean()
        print(f"\n  [Q2.2] Mean entropy - disordered {dis_mean:.3f}, "
              f"structured {str_mean:.3f}")
        if dis_mean > str_mean:
            print(f"         Disordered residues have MORE diffuse attention "
                  f"(+{dis_mean-str_mean:.3f})")
        else:
            print(f"         Structured residues have more diffuse attention "
                  f"(+{str_mean-dis_mean:.3f})")

    # ── Q2.3 aggregate ──
    if flow_cnt_total.sum() > 0:
        # Average cross-boundary fraction per offset (over layers/heads too)
        # flow_sum_total: [33 x 20 x n_off], divide by counts per offset
        safe_cnt = np.maximum(flow_cnt_total, 1)
        flow_avg = flow_sum_total / safe_cnt[None, None, :]   # [33 x 20 x n_off]
        results["cross_boundary_flow"] = flow_avg
        results["boundary_offsets"]    = offsets_global
        results["boundary_counts"]     = flow_cnt_total

        # Summary: averaged over all layers/heads, profile vs offset
        profile = flow_avg.mean(axis=(0, 1))   # [n_off]
        mid = len(profile) // 2
        print(f"\n  [Q2.3] Cross-boundary flow profile (avg over layers/heads):")
        print(f"         at boundary (offset 0): {profile[mid]:.3f}")
        print(f"         far left (-{BOUNDARY_WIN}):  {profile[0]:.3f}")
        print(f"         far right (+{BOUNDARY_WIN}): {profile[-1]:.3f}")

    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(OUT_DIR, exist_ok=True)

    for binding_type in BINDING_TYPES:
        save_path = os.path.join(OUT_DIR, f"{binding_type}_q2_results.npz")
        if os.path.exists(save_path):
            print(f"\n  Skipping {binding_type} - already done")
            continue

        results = analyze_binding_type(binding_type, device)
        if results:
            np.savez(save_path, **{k: v for k, v in results.items()
                                   if isinstance(v, np.ndarray)})
            print(f"  Saved: {save_path}")

    print(f"\n Q2 analysis complete. Results in {OUT_DIR}")


if __name__ == "__main__":
    main()