"""
Q1 Figures
==========
Generates publication-quality figures from Q1 analysis results.

Figures produced:
  1. Enrichment heatmaps [33 layers x 20 heads] per binding type
  2. Within-site heatmaps [33 layers x 20 heads] per binding type
  3. Layer curves comparing all binding types
  4. Combined summary figure

Run on login node (no GPU needed).

Usage:
  python q1_figures.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_DIR = "/work/malekia/esm2-idp-interpretability/outputs/q1_analysis"
FIG_DIR     = "/work/malekia/esm2-idp-interpretability/outputs/q1_figures"

BINDING_TYPES = ["ion", "protein", "dna", "rna"]
TYPE_COLORS   = {"ion": "#E63946", "protein": "#457B9D",
                 "dna": "#2A9D8F", "rna": "#E9C46A"}
TYPE_LABELS   = {"ion": "Ion", "protein": "Protein",
                 "dna": "DNA", "rna": "RNA"}


def load_results(binding_type):
    path = os.path.join(RESULTS_DIR, f"{binding_type}_q1_results.npz")
    if not os.path.exists(path):
        return None
    return dict(np.load(path))


# ─────────────────────────────────────────────
# FIGURE 1 — Enrichment heatmaps
# ─────────────────────────────────────────────
def fig_enrichment_heatmaps(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Attention Enrichment at Binding Sites  [layers × heads]",
                 fontsize=15, fontweight='bold')

    for ax, btype in zip(axes.flat, BINDING_TYPES):
        res = all_results.get(btype)
        if res is None or "mean_enrichment" not in res:
            ax.set_visible(False)
            continue

        data = res["mean_enrichment"]   # [33 x 20]

        im = ax.imshow(data, aspect='auto', cmap='RdBu_r',
                       vmin=0.5, vmax=1.5, origin='lower')
        ax.set_title(f"{TYPE_LABELS[btype]} binding", fontweight='bold')
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_xticks(range(0, 20, 2))
        ax.set_yticks(range(0, 33, 4))
        ax.set_yticklabels(range(1, 34, 4))
        plt.colorbar(im, ax=ax, label="Enrichment (x random)")

        # Mark the best cell
        best_idx = np.unravel_index(data.argmax(), data.shape)
        ax.plot(best_idx[1], best_idx[0], 'k*', markersize=15)

    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "fig1_enrichment_heatmaps.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────
# FIGURE 2 — Within-site heatmaps
# ─────────────────────────────────────────────
def fig_within_heatmaps(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Within-Binding-Site Attention  [layers × heads]",
                 fontsize=15, fontweight='bold')

    for ax, btype in zip(axes.flat, BINDING_TYPES):
        res = all_results.get(btype)
        if res is None or "mean_within" not in res:
            ax.set_visible(False)
            continue

        data = res["mean_within"]   # [33 x 20]
        baseline = res["mean_baseline"][0] if "mean_baseline" in res else 0

        im = ax.imshow(data, aspect='auto', cmap='viridis',
                       vmin=0, vmax=1, origin='lower')
        ax.set_title(f"{TYPE_LABELS[btype]} binding (baseline={baseline:.2f})",
                     fontweight='bold')
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_xticks(range(0, 20, 2))
        ax.set_yticks(range(0, 33, 4))
        ax.set_yticklabels(range(1, 34, 4))
        plt.colorbar(im, ax=ax, label="Within-site fraction")

        best_idx = np.unravel_index(data.argmax(), data.shape)
        ax.plot(best_idx[1], best_idx[0], 'r*', markersize=15)

    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "fig2_within_site_heatmaps.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────
# FIGURE 3 — Layer curves (all types overlaid)
# ─────────────────────────────────────────────
def fig_layer_curves(all_results):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: enrichment per layer (max across heads)
    ax = axes[0]
    for btype in BINDING_TYPES:
        res = all_results.get(btype)
        if res is None or "mean_enrichment" not in res:
            continue
        layer_max = res["mean_enrichment"].max(axis=1)   # [33]
        ax.plot(range(1, 34), layer_max, marker='o', markersize=4,
                label=TYPE_LABELS[btype], color=TYPE_COLORS[btype], linewidth=2)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Enrichment (max across heads)", fontsize=12)
    ax.set_title("Attention Enrichment by Layer", fontweight='bold', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: within-site per layer (max across heads)
    ax = axes[1]
    for btype in BINDING_TYPES:
        res = all_results.get(btype)
        if res is None or "mean_within" not in res:
            continue
        layer_max = res["mean_within"].max(axis=1)   # [33]
        ax.plot(range(1, 34), layer_max, marker='o', markersize=4,
                label=TYPE_LABELS[btype], color=TYPE_COLORS[btype], linewidth=2)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Within-site fraction (max across heads)", fontsize=12)
    ax.set_title("Within-Site Attention by Layer", fontweight='bold', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "fig3_layer_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    all_results = {}
    for btype in BINDING_TYPES:
        res = load_results(btype)
        if res is not None:
            all_results[btype] = res
            print(f"Loaded {btype} results")

    if not all_results:
        print("No results found! Run q1_attention_analysis.py first.")
        return

    print()
    fig_enrichment_heatmaps(all_results)
    fig_within_heatmaps(all_results)
    fig_layer_curves(all_results)

    print(f"\n All figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()