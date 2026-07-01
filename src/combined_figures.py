"""
Combined Disordered vs Structured Figures (Q1 and Q3)
======================================================
Overlays disordered and structured results on the same axes so the
depth difference (disordered peaks early, structured peaks late) is
directly visible.

Q1: attention enrichment by layer, disordered vs structured
Q3: probing AUPRC by layer, disordered vs structured

Disordered = solid line, Structured = dashed line, one color per binding type.

Usage:
  python combined_figures.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

BASE = "/work/malekia/esm2-idp-interpretability/outputs"

# Result directories
Q1_DIS = os.path.join(BASE, "q1_analysis")
Q1_STR = os.path.join(BASE, "q1_analysis_structured")
Q3_DIS = os.path.join(BASE, "q3_analysis")
Q3_STR = os.path.join(BASE, "q3_analysis_structured")

FIG_DIR = os.path.join(BASE, "combined_figures")

BINDING_TYPES = ["ion", "protein", "dna", "rna"]
TYPE_COLORS = {"ion": "#E63946", "protein": "#457B9D",
               "dna": "#2A9D8F", "rna": "#E9C46A"}
TYPE_LABELS = {"ion": "Ion", "protein": "Protein", "dna": "DNA", "rna": "RNA"}


def load(path, btype, suffix):
    f = os.path.join(path, f"{btype}_{suffix}.npz")
    if not os.path.exists(f):
        return None
    return dict(np.load(f))


# ─────────────────────────────────────────────
# Q1 - enrichment by layer, disordered vs structured
# ─────────────────────────────────────────────
def fig_q1_combined():
    fig, ax = plt.subplots(figsize=(12, 7))

    for btype in BINDING_TYPES:
        dis = load(Q1_DIS, btype, "q1_results")
        strc = load(Q1_STR, btype, "q1_results")
        color = TYPE_COLORS[btype]

        if dis is not None and "mean_enrichment" in dis:
            layer_max = dis["mean_enrichment"].max(axis=1)   # [33]
            ax.plot(range(1, 34), layer_max, '-', color=color, linewidth=2,
                    label=f"{TYPE_LABELS[btype]} (disordered)")

        if strc is not None and "mean_enrichment" in strc:
            layer_max = strc["mean_enrichment"].max(axis=1)
            ax.plot(range(1, 34), layer_max, '--', color=color, linewidth=2,
                    label=f"{TYPE_LABELS[btype]} (structured)")

    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Attention enrichment (max across heads)", fontsize=12)
    ax.set_title("Q1: Attention Enrichment by Layer - Disordered vs Structured\n"
                 "(solid = disordered, dashed = structured)",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_xticks(range(0, 34, 2))

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "q1_disordered_vs_structured.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
# Q3 - AUPRC by layer, disordered vs structured
# ─────────────────────────────────────────────
def fig_q3_combined():
    # 2x2 grid, one panel per binding type (clearer than 8 lines on one axis)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Q3: Probing AUPRC by Layer - Disordered vs Structured\n"
                 "(solid = disordered, dashed = structured)",
                 fontsize=14, fontweight='bold')

    for ax, btype in zip(axes.flat, BINDING_TYPES):
        color = TYPE_COLORS[btype]
        dis = load(Q3_DIS, btype, "q3_results")
        strc = load(Q3_STR, btype, "q3_results")

        if dis is not None:
            ax.plot(dis["layers"], dis["auprc"], '-', color=color,
                    linewidth=2, marker='o', markersize=3, label="Disordered")
        if strc is not None:
            ax.plot(strc["layers"], strc["auprc"], '--', color=color,
                    linewidth=2, marker='s', markersize=3, label="Structured")

        # Mark best layer for each
        if dis is not None:
            bi = int(np.argmax(dis["auprc"]))
            ax.axvline(dis["layers"][bi], color=color, linestyle='-', alpha=0.25)
        if strc is not None:
            bi = int(np.argmax(strc["auprc"]))
            ax.axvline(strc["layers"][bi], color=color, linestyle='--', alpha=0.25)

        ax.set_title(f"{TYPE_LABELS[btype]} binding", fontweight='bold')
        ax.set_xlabel("Layer")
        ax.set_ylabel("AUPRC")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "q3_disordered_vs_structured.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_q1_combined()
    fig_q3_combined()
    print(f"\n Combined figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()