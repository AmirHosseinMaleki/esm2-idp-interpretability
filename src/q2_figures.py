"""
Q2 Figures
==========
Generates figures from Q2 disorder analysis results.

Figures:
  1. Disorder-preference heatmaps [layer x head] per binding type (Q2.1)
  2. Disordered vs structured entropy comparison (Q2.2)
  3. Cross-boundary flow profile vs distance from boundary (Q2.3) — key figure

Usage:
  python q2_figures.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "/work/malekia/esm2-idp-interpretability/outputs/q2_analysis"
FIG_DIR     = "/work/malekia/esm2-idp-interpretability/outputs/q2_figures"

BINDING_TYPES = ["ion", "protein", "dna", "rna"]
TYPE_COLORS   = {"ion": "#E63946", "protein": "#457B9D",
                 "dna": "#2A9D8F", "rna": "#E9C46A"}
TYPE_LABELS   = {"ion": "Ion", "protein": "Protein",
                 "dna": "DNA", "rna": "RNA"}


def load_results(binding_type):
    path = os.path.join(RESULTS_DIR, f"{binding_type}_q2_results.npz")
    if not os.path.exists(path):
        return None
    return dict(np.load(path))


# ─────────────────────────────────────────────
# FIGURE 1 — Q2.1 disorder preference heatmaps
# ─────────────────────────────────────────────
def fig_disorder_preference(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Q2.1: Attention Preference for Disordered Residues  [layer × head]\n"
                 "(>1 = head attends more to disordered than structured residues)",
                 fontsize=14, fontweight='bold')

    for ax, btype in zip(axes.flat, BINDING_TYPES):
        res = all_results.get(btype)
        if res is None or "disorder_preference" not in res:
            ax.set_visible(False)
            continue

        data = res["disorder_preference"]   # [33 x 20]
        # Use log scale centered at 1 so >1 and <1 are symmetric
        im = ax.imshow(data, aspect='auto', cmap='RdBu_r',
                       vmin=0, vmax=3, origin='lower')
        ax.set_title(f"{TYPE_LABELS[btype]} binding", fontweight='bold')
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_xticks(range(0, 20, 2))
        ax.set_yticks(range(0, 33, 4))
        ax.set_yticklabels(range(1, 34, 4))
        plt.colorbar(im, ax=ax, label="Disorder preference (×)")

        best = np.unravel_index(data.argmax(), data.shape)
        ax.plot(best[1], best[0], 'k*', markersize=16)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig1_disorder_preference.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
# FIGURE 2 — Q2.2 entropy comparison
# ─────────────────────────────────────────────
def fig_entropy(all_results):
    fig, ax = plt.subplots(figsize=(10, 6))

    types = []
    dis_means = []
    str_means = []
    colors = []

    for btype in BINDING_TYPES:
        res = all_results.get(btype)
        if res is None or "disordered_entropy" not in res:
            continue
        types.append(TYPE_LABELS[btype])
        dis_means.append(res["disordered_entropy"].mean())
        str_means.append(res["structured_entropy"].mean())
        colors.append(TYPE_COLORS[btype])

    x = np.arange(len(types))
    width = 0.35

    ax.bar(x - width/2, dis_means, width, label='Disordered',
           color=colors, edgecolor='black')
    ax.bar(x + width/2, str_means, width, label='Structured',
           color=colors, alpha=0.45, edgecolor='black')

    ax.set_ylabel("Mean attention entropy", fontsize=12)
    ax.set_title("Q2.2: Attention Entropy — Disordered vs Structured Residues\n"
                 "(higher = more diffuse attention)",
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.legend()
    # Zoom y-axis to make the consistent difference visible
    all_vals = dis_means + str_means
    ax.set_ylim(min(all_vals) - 0.15, max(all_vals) + 0.1)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig2_entropy.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
# FIGURE 3 — Q2.3 cross-boundary profile (KEY FIGURE)
# ─────────────────────────────────────────────
def fig_boundary_profile(all_results):
    fig, ax = plt.subplots(figsize=(11, 7))

    for btype in BINDING_TYPES:
        res = all_results.get(btype)
        if res is None or "cross_boundary_flow" not in res:
            continue

        flow = res["cross_boundary_flow"]        # [33 x 20 x n_off]
        offsets = res["boundary_offsets"]        # [n_off]
        profile = flow.mean(axis=(0, 1))         # average over layers/heads

        ax.plot(offsets, profile, marker='o', markersize=3,
                label=TYPE_LABELS[btype], color=TYPE_COLORS[btype],
                linewidth=2)

    ax.axvline(0, color='gray', linestyle='--', alpha=0.7, label='Boundary')
    ax.set_xlabel("Distance from disorder/structure boundary (residues)", fontsize=12)
    ax.set_ylabel("Fraction of attention crossing the boundary", fontsize=12)
    ax.set_title("Q2.3: Cross-Boundary Attention at the Disorder/Structure Edge\n"
                 "(peaks at the boundary — residues at the edge integrate both sides)",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig3_boundary_profile.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
# FIGURE 4 — Q2.1 disorder preference by layer (line summary)
# ─────────────────────────────────────────────
def fig_preference_by_layer(all_results):
    fig, ax = plt.subplots(figsize=(11, 6))

    for btype in BINDING_TYPES:
        res = all_results.get(btype)
        if res is None or "disorder_preference" not in res:
            continue
        layer_max = res["disorder_preference"].max(axis=1)   # [33]
        ax.plot(range(1, 34), layer_max, marker='o', markersize=4,
                label=TYPE_LABELS[btype], color=TYPE_COLORS[btype], linewidth=2)

    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='No preference')
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Max disorder preference across heads (×)", fontsize=12)
    ax.set_title("Q2.1: Disorder Preference by Layer",
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(range(0, 34, 2))

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig4_preference_by_layer.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    all_results = {}
    for btype in BINDING_TYPES:
        res = load_results(btype)
        if res is not None:
            all_results[btype] = res
            print(f"Loaded {btype}")

    if not all_results:
        print("No results found! Run q2_disorder_analysis.py first.")
        return

    print()
    fig_disorder_preference(all_results)
    fig_preference_by_layer(all_results)
    fig_entropy(all_results)
    fig_boundary_profile(all_results)

    print(f"\n Q2 figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()