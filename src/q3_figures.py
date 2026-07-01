"""
Q3 Figures
==========
Generates layer-probing curves from Q3 results.
Primary metrics: AUPRC and MCC (robust to class imbalance).
AUROC shown as secondary for completeness.

Usage:
  python q3_figures.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "/work/malekia/esm2-idp-interpretability/outputs/q3_analysis"
FIG_DIR     = "/work/malekia/esm2-idp-interpretability/outputs/q3_figures"

BINDING_TYPES = ["ion", "protein", "dna", "rna"]
TYPE_COLORS   = {"ion": "#E63946", "protein": "#457B9D",
                 "dna": "#2A9D8F", "rna": "#E9C46A"}
TYPE_LABELS   = {"ion": "Ion", "protein": "Protein",
                 "dna": "DNA", "rna": "RNA"}


def load_results(binding_type):
    path = os.path.join(RESULTS_DIR, f"{binding_type}_q3_results.npz")
    if not os.path.exists(path):
        return None
    return dict(np.load(path))


# ─────────────────────────────────────────────
# FIGURE 1 — All metrics, all types (the main figure)
# ─────────────────────────────────────────────
def fig_all_metrics(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Q3: Binding Site Information Across ESM2 Layers",
                 fontsize=16, fontweight='bold')

    metrics = [
        ("auprc", "AUPRC (primary — robust to imbalance)", axes[0, 0]),
        ("mcc",   "MCC (primary — balanced)",              axes[0, 1]),
        ("auroc", "AUROC (secondary — inflated by negatives)", axes[1, 0]),
        ("f1",    "F1 (threshold-dependent)",              axes[1, 1]),
    ]

    for metric_key, title, ax in metrics:
        for btype in BINDING_TYPES:
            res = all_results.get(btype)
            if res is None:
                continue
            layers = res["layers"]
            values = res[metric_key]
            ax.plot(layers, values, marker='o', markersize=4,
                    label=TYPE_LABELS[btype], color=TYPE_COLORS[btype],
                    linewidth=2)

            # Mark best layer with a star
            best_idx = np.argmax(values)
            ax.plot(layers[best_idx], values[best_idx], '*',
                    color=TYPE_COLORS[btype], markersize=18,
                    markeredgecolor='black', markeredgewidth=0.5)

        ax.set_xlabel("ESM2 Layer", fontsize=12)
        ax.set_ylabel(metric_key.upper(), fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        ax.set_xticks(range(0, 34, 4))

    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "fig1_all_metrics.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────
# FIGURE 2 — AUPRC only (clean thesis figure)
# ─────────────────────────────────────────────
def fig_auprc_clean(all_results):
    fig, ax = plt.subplots(figsize=(11, 7))

    for btype in BINDING_TYPES:
        res = all_results.get(btype)
        if res is None:
            continue
        layers = res["layers"]
        values = res["auprc"]
        ax.plot(layers, values, marker='o', markersize=5,
                label=TYPE_LABELS[btype], color=TYPE_COLORS[btype],
                linewidth=2.5)

        best_idx = np.argmax(values)
        best_layer = layers[best_idx]
        ax.plot(best_layer, values[best_idx], '*',
                color=TYPE_COLORS[btype], markersize=22,
                markeredgecolor='black', markeredgewidth=1)
        # Annotate best layer
        ax.annotate(f"L{best_layer}",
                    (best_layer, values[best_idx]),
                    textcoords="offset points", xytext=(0, 12),
                    ha='center', fontsize=11, fontweight='bold',
                    color=TYPE_COLORS[btype])

    ax.set_xlabel("ESM2 Layer", fontsize=13)
    ax.set_ylabel("AUPRC", fontsize=13)
    ax.set_title("Binding Site Information by Layer (AUPRC)\n"
                 "Stars mark the best layer per binding type",
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xticks(range(0, 34, 2))

    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "fig2_auprc_clean.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────
# FIGURE 3 — Best layer comparison bar chart
# ─────────────────────────────────────────────
def fig_best_layers(all_results):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    types = []
    best_layers = []
    best_auprc = []
    last_auprc = []
    colors = []

    for btype in BINDING_TYPES:
        res = all_results.get(btype)
        if res is None:
            continue
        values = res["auprc"]
        best_idx = np.argmax(values)
        types.append(TYPE_LABELS[btype])
        best_layers.append(res["layers"][best_idx])
        best_auprc.append(values[best_idx])
        last_auprc.append(values[-1])
        colors.append(TYPE_COLORS[btype])

    # Left: best layer per type
    ax = axes[0]
    bars = ax.bar(types, best_layers, color=colors, edgecolor='black')
    ax.set_ylabel("Best Layer", fontsize=12)
    ax.set_title("Optimal Layer per Binding Type (by AUPRC)",
                 fontsize=13, fontweight='bold')
    ax.axhline(33, color='gray', linestyle='--', alpha=0.7,
               label='Last layer (33)')
    ax.legend()
    for bar, layer in zip(bars, best_layers):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"L{layer}", ha='center', fontweight='bold')

    # Right: best vs last layer AUPRC
    ax = axes[1]
    x = np.arange(len(types))
    width = 0.35
    ax.bar(x - width/2, best_auprc, width, label='Best layer',
           color=colors, edgecolor='black')
    ax.bar(x + width/2, last_auprc, width, label='Last layer (33)',
           color=colors, alpha=0.4, edgecolor='black')
    ax.set_ylabel("AUPRC", fontsize=12)
    ax.set_title("Best Layer vs Last Layer Performance",
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "fig3_best_layers.png")
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
            print(f"Loaded {btype}")

    if not all_results:
        print("No results found! Run q3_probing.py first.")
        return

    print()
    fig_all_metrics(all_results)
    fig_auprc_clean(all_results)
    fig_best_layers(all_results)

    # Print summary table with all metrics
    print(f"\n{'='*70}")
    print(f"  BEST LAYER BY EACH METRIC (robustness check)")
    print(f"{'='*70}")
    print(f"  {'Type':<10} {'AUPRC':>14} {'MCC':>14} {'AUROC':>14}")
    print(f"  {'-'*54}")
    for btype in BINDING_TYPES:
        res = all_results.get(btype)
        if res is None:
            continue
        best_auprc = res['layers'][np.argmax(res['auprc'])]
        best_mcc   = res['layers'][np.argmax(res['mcc'])]
        best_auroc = res['layers'][np.argmax(res['auroc'])]
        agree = "✓ agree" if best_auprc == best_mcc == best_auroc else "differ"
        print(f"  {TYPE_LABELS[btype]:<10} "
              f"L{best_auprc:<13} L{best_mcc:<13} L{best_auroc:<13} {agree}")

    print(f"\n Q3 figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()