# Interpretability Analysis of ESM2 Representations for IDP Binding Sites
### Full Progress Report

This report covers the complete analysis for all three research questions, on both **disordered** (DisProt) and **structured** (AHoJ / ScanNet / BioLiP) proteins. Q1 (attention at binding sites) and Q3 (layer probing) are done for both data types; Q2 (attention in disordered vs structured regions) is complete.

---

## Pipeline and data

I ran ESM2 (650M, 33 layers) once per protein and saved, per protein, the attention tensors `[33 × 20 × L × L]` and embeddings from all 34 layers `[34 × L × 1280]`. All later analyses read from these saved files.

- **Disordered (DisProt):** 991/992 proteins processed (one, L = 34,350, is too long for a single GPU) - ion, protein, DNA, RNA binding.
- **Structured:** 12,428 proteins - ion (AHoJ-DB), protein (ScanNet), DNA and RNA (BioLiP).

Two data issues were found and fixed for the structured set. The AHoJ ion data stored one row per ligand-binding event rather than per protein chain, so I merged annotations per chain (using OR across events) before clustering. ScanNet protein was sampled to 2,000 proteins to keep the analysis tractable. All structured sets were clustered at 10% identity to prevent train/test leakage.

For Q2 I fetched per-residue **disorder annotations** directly from DisProt (1 = disordered, 0 = structured) for the IDP proteins, stored separately from the binding data. Of ~970 proteins, **720 are "mosaic"** - they contain both disordered and structured regions, which is what the boundary analysis needs.

---

## Q1 - Attention at binding sites

*Research question: How does attention behave at known binding sites? Do binding site residues attend to each other?*

**How I did it:** For every protein, layer, and head, I computed two things from the saved attention tensors: **enrichment** - how much more attention binding site residues receive than the average residue - and **within-site attention** - the fraction of attention sent by binding site residues that goes to other binding site residues, versus a random baseline. I averaged across all proteins per binding type. Proteins where every residue is a binding site are excluded (enrichment is undefined there).

### Disordered

| Binding type | Enrichment (best layer) | Within-site attention (vs random) |
|---|---|---|
| Ion | 1.52× (layer 8) | 3.2× (layer 32) |
| Protein | 1.49× (layer 8) | 3.5× (layer 32) |
| DNA | 1.46× (layer 3) | 3.2× (layer 3) |
| RNA | 1.75× (layer 29) | 3.0× (layer 3) |

Binding site residues attract more attention than random (1.4–1.7×) - a modest but consistent effect - and they attend strongly to each other (~3× over random), peaking in the final layers.

### Structured vs disordered (enrichment best layer)

| Type | Disordered | Structured |
|---|---|---|
| Ion | 1.52× (layer 8) | 5.11× (layer 17) |
| Protein | 1.49× (layer 8) | 1.14× (layer 31) |
| DNA | 1.46× (layer 3) | 2.10× (layer 33) |
| RNA | 1.75× (layer 29) | 1.95× (layer 31) |

Structured binding sites generally show a stronger and later attention signal than disordered ones (ion especially, 5.11× vs 1.52×) - consistent with structured sites depending on the folded 3D context that ESM2 builds in deeper layers. Structured protein binding is the exception, with weak enrichment (1.14×), reflecting how diffuse protein-protein interfaces are.

![Q1 enrichment by layer: disordered vs structured](outputs/combined_figures/q1_disordered_vs_structured.png)

**Caveat:** the within-site metric is partly inflated by binding sites being contiguous stretches combined with attention's local bias. The "×over random" baseline controls for the number of binding sites but not for contiguity - a follow-up control is planned.

---

## Q3 - Which layer best encodes binding information

*Research question: How does binding information evolve across ESM2's layers? Does the last layer encode it best?*

**How I did it:** For each binding type and each of the 34 layers, I trained a simple logistic-regression probe to predict per-residue binding from that layer's embeddings, using the existing train/test split. A simple probe means a good score reflects what the embedding encodes, not the probe's power. I used AUPRC and MCC as primary metrics (robust to the rare-positive imbalance), with AUROC secondary.

### Best layer by AUPRC

| Type | Disordered | Structured |
|---|---|---|
| Ion | layer 5 (0.560) | layer 31 (0.422) |
| Protein | layer 33 (0.604) | layer 33 (0.425) |
| DNA | layer 4 (0.625) | layer 31 (0.399) |
| RNA | layer 32 (0.455) | layer 33 (0.430) |

This is the clearest result. For **disordered** ion and DNA binding, information peaks in early layers (4–5); for **structured** binding of the same types, it peaks late (31). Disorder makes binding sites readable earlier in the network, because there is no structural context to integrate; structured sites require the full depth. The last layer is not universally best - only protein binding peaks at the final layer in the disordered set.

![Q3 AUPRC by layer: disordered vs structured](outputs/combined_figures/q3_disordered_vs_structured.png)

**Note on absolute values:** structured AUPRC is lower than disordered (e.g. ion 0.42 vs 0.56), but this mostly reflects the much lower positive-class density in the structured data (ion ~2.3% binding vs ~31% in DisProt). The robust comparison is the *layer pattern* (where the peak is), not the absolute height.

Q1 (attention) and Q3 (probing) measure different things - Q1 looks at each layer's attention, Q3 at each layer's embeddings - so their exact best layers are not expected to match, and Q1's two metrics themselves often peak at different depths. But the broad depth trend agrees: DNA binding peaks early in both (Q1 enrichment layer 3, Q3 layer 4), while protein binding peaks late in both (Q1 within-site layer 32, Q3 layer 33). That two independent methods point the same way strengthens the finding.

---

## Q2 - Attention in disordered vs structured regions

*Research question: Which layers attend to IDP regions? How does attention behave differently in disordered vs structured regions? How does it behave at the edge (boundary)?*

**How I did it:** Using the DisProt disorder annotations, I ran three analyses on the 720 mosaic proteins - proteins with reliable per-residue disorder labels that contain both disordered and structured regions - each comparing the two region types *within* the same protein (which controls for protein-specific effects). This within-protein design is why the separate structured dataset is not needed for Q2: both region types are already present, and labeled, inside these proteins. **Q2.1** measures attention received by disordered vs structured residues. **Q2.2** measures attention entropy (how focused vs diffuse a residue's attention is). **Q2.3** measures cross-boundary flow - the fraction of a residue's attention that crosses the disorder/structure boundary, as a function of distance from it.

### Q2.1 - Which layers attend to disordered regions

| Type | Best layer | Disorder preference |
|---|---|---|
| Ion | 12 | 10.09× |
| Protein | 2 | 2.31× |
| DNA | 22 | 2.67× |
| RNA | 22 | 10.52× |

Specific heads attend up to 10× more to disordered residues than to structured ones. ESM2 clearly distinguishes disordered regions, though the depth at which it does so varies by binding type.

![Q2.1 disorder preference](outputs/q2_figures/fig1_disorder_preference.png)

### Q2.2 - How attention behaves (entropy)

| Type | Disordered entropy | Structured entropy | Difference |
|---|---|---|---|
| Ion | 3.846 | 3.794 | +0.052 |
| Protein | 4.062 | 3.922 | +0.140 |
| DNA | 3.993 | 3.873 | +0.120 |
| RNA | 4.161 | 4.044 | +0.117 |

Disordered residues have consistently more diffuse attention (higher entropy) than structured residues, in all four binding types. The effect is modest but perfectly consistent in direction - structured residues attend in a more focused way (they have specific structural contacts), while disordered residues spread their attention more (they are flexible, with no fixed partners).

![Q2.2 entropy comparison](outputs/q2_figures/fig2_entropy.png)

### Q2.3 - Behavior at the edge (boundary)

| Type | At boundary (offset 0) | Far from boundary (±20) |
|---|---|---|
| Ion | 0.465 | ~0.26 |
| Protein | 0.480 | ~0.27 |
| DNA | 0.487 | ~0.27 |
| RNA | 0.489 | ~0.28 |

Cross-boundary attention peaks sharply at the boundary and drops off into either region - again identical across all four types. Residues right at the disorder/structure edge send ~48% of their attention across to the other side; residues deep within a region send only ~27%. Boundary residues integrate information from both sides, which is exactly the "edge" behavior the research question asks about.

![Q2.3 boundary profile (key figure)](outputs/q2_figures/fig3_boundary_profile.png)

---

## What I take from this

The central theme, consistent across both Q1 and Q3, is that **binding information lives at different network depths depending on both the binding type and the structural context.** Disordered ion and DNA binding, which rely on local sequence chemistry, are captured early; structured binding of the same types needs the full depth to integrate 3D context. Two independent methods - attention (Q1) and probing (Q3) - agree on this, which strengthens it.

Q2 adds a third, consistent picture: ESM2 not only distinguishes disordered from structured residues (up to 10× attention preference, more diffuse attention on disordered regions), it also treats the boundary between them as special, concentrating cross-region attention right at the edge. All three Q2 metrics give the same answer across all four binding types.

One practical implication runs through the whole analysis: the research project used last-layer embeddings throughout, but these results suggest the optimal layer depends on binding type and structural context - for disordered ion/DNA, layers 4–5 clearly beat the last layer.

---

## Methodological notes

- AUROC is inflated on imbalanced binding data, so AUPRC and MCC are the primary metrics; all three agree on the best layer for every type.
- Q2 uses real DisProt disorder annotations, not a predictor - DisProt curates disorder regions directly.
- For the boundary window, rather than picking an arbitrary size I analyze attention across the full profile (−20 to +20) and read the transition width from it; summary stats use ±5 (the scale of disordered binding motifs).
- Q2 within-protein comparison (disordered vs structured residues in the same chain) controls for protein-level confounds.
- Proteins longer than 1,800 residues were excluded from Q2 (40 of 720, ~5.6%) due to the memory cost of the entropy computation; this does not bias the result, since length is unrelated to the disorder question.

---

## Next steps

- (Optional) Compare structured regions in IDPs against structured regions in fully structured proteins, to check whether structural context differs between the two - a possible extension, not a gap.
- Q1 second part: compare correctly-predicted vs missed binding sites (needs predictions from the research-project model), and add a contiguity control for the within-site metric.
- Regenerate results uniformly and integrate all three questions into the thesis write-up.
- Continue the literature review on transformer interpretability.