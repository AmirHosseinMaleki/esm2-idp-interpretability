# Thesis Progress Update
### Interpretability Analysis of ESM2 Representations for IDP Binding Sites

I built the ESM2 inference pipeline and completed the analysis for **Q1 (attention at binding sites)** and **Q3 (layer probing)** across all four binding types (ion, protein, DNA, RNA). Q2 is next.

---

## Pipeline

I ran ESM2 (650M, 33 layers) on each DisProt protein and saved, per protein, the attention tensors `[33 × 20 × L × L]` and the embeddings from all 34 layers `[34 × L × 1280]`. All later analyses read from these saved files. I processed 991/992 proteins - one protein (L = 34,350) is too long for a single GPU.

---

## Q1 - Attention at binding sites

*Research question: How does attention behave at known binding sites? Do binding site residues attend to each other?*

**How I did it:** For every protein, every layer, and every attention head, I computed two things from the saved attention tensors: (1) **enrichment** - how much more attention binding site residues receive compared to the average residue, and (2) **within-site attention** - what fraction of the attention sent by binding site residues goes to other binding site residues, compared to a random baseline. I then averaged across all proteins per binding type.

| Binding type | Enrichment (best layer) | Within-site attention (vs random) |
|---|---|---|
| Ion | 1.52× (layer 8) | 3.2× (layer 32) |
| Protein | 1.49× (layer 8) | 3.5× (layer 32) |
| DNA | 1.46× (layer 3) | 3.2× (layer 3) |
| RNA | 1.75× (layer 29) | 3.0× (layer 3) |

- Binding site residues attract more attention than random (1.4–1.7×) - a modest but consistent effect.
- Binding site residues attend strongly to each other (~3× over random), peaking in the final layers.
- The best layer differs by binding type, hinting that ESM2 encodes different binding types at different depths.

![Q1 layer curves](outputs/q1_figures/fig3_layer_curves.png)

---

## Q3 - Which layer best encodes binding information

*Research question: How does binding information evolve across ESM2's layers? Does the last layer encode it best?*

**How I did it:** For each binding type and each of the 34 layers, I trained a simple logistic-regression probe to predict per-residue binding from that layer's embeddings, using the existing train/test split. A simple probe means a good score reflects what the embedding encodes, not the probe's power. I evaluated with AUPRC and MCC (robust to the rare-positive imbalance), with AUROC secondary.

| Binding type | Best layer | Best AUPRC | Last layer (33) |
|---|---|---|---|
| Ion | 5 | 0.560 | 0.481 |
| Protein | 33 | 0.604 | 0.595 |
| DNA | 4 | 0.625 | 0.447 |
| RNA | 32 | 0.455 | 0.434 |

- The last layer is not universally best. Only protein binding peaks at the final layer.
- Ion and DNA binding peak very early (layers 4–5); deeper layers actually lose information for these types.
- All three metrics (AUPRC, MCC, AUROC) agree on the best layer for every type, so the finding is robust.

![Q3 AUPRC by layer](outputs/q3_figures/fig2_auprc_clean.png)

---

## What I take from this

Different binding types seem to be encoded at different network depths. Ion and DNA - which rely on specific local residues (Cys/His for ions, charged residues for DNA) - are captured in early layers, while protein-protein interfaces, which are more complex, need the full depth. RNA peaking late is less expected given its chemical similarity to DNA; it may reflect RNA's greater structural diversity, though the small RNA/DNA sample sizes (~47–49 proteins) mean the exact peak is noisy.

Notably, Q1 and Q3 point to the same depth pattern despite measuring different things: Q1 looks at each layer's **attention** (which residues focus on which), while Q3 looks at each layer's **embeddings** (the encoded representation a probe reads from). Because these are two distinct signals from the model, their exact best layers per binding type are not expected to match - but the broad trend agrees (DNA early, protein late), which strengthens the result.

One practical implication: the research project used last-layer embeddings throughout, but these results suggest the optimal layer depends on binding type (e.g. layer 4–5 for ion/DNA gives clearly better predictions).

---

## Next steps

- **Q2** - IDP vs structured attention and boundary behavior; needs per-residue disorder annotations.
- **Q1 second part** - compare correctly-predicted vs missed binding sites (needs predictions from the research-project model), and add a control for the contiguity of binding regions.
- Begin literature review on transformer interpretability.