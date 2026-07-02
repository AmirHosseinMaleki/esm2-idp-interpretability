# Interpretability Analysis of ESM2 Representations for IDP Binding Sites

This report covers the complete analysis for all three research questions, on both **disordered** (DisProt) and **structured** (AHoJ / ScanNet / BioLiP) proteins. Q1 (attention at binding sites) and Q3 (layer probing) are done for both data types; Q2 (attention in disordered vs structured regions) is complete.

---

## Pipeline and data

I ran ESM2 (650M, 33 layers) once per protein and saved, per protein, the attention tensors `[33 × 20 × L × L]` and embeddings from all 34 layers `[34 × L × 1280]`. All later analyses read from these saved files.

- **Disordered (DisProt):** 991/992 proteins processed (one, L = 34,350, is too long for a single GPU) - ion, protein, DNA, RNA binding.
- **Structured:** 12,428 proteins - ion (AHoJ-DB), protein (ScanNet), DNA and RNA (BioLiP).

For Q2 I fetched per-residue **disorder annotations** directly from DisProt (1 = disordered, 0 = structured) for the IDP proteins, stored separately from the binding data. Of ~970 proteins, **720 are "mosaic"** - they contain both disordered and structured regions, which is what the boundary analysis needs.

---

## Metrics: what they measure and how they are computed

Before the results, here is a plain definition of every metric used, the formula, and how to read it. Attention in ESM2 is a set of matrices `A`, one per layer (33) and head (20). Each entry `A[i, j]` is how much residue `i` attends to residue `j`. Every row sums to 1 (each residue distributes a total "budget" of 1 across all residues).

**Attention received (used in Q1 enrichment, Q2.1).**
For a residue `j`, this is the total attention it gets from everyone: the sum down column `j` of the matrix (`sum over i of A[i, j]`). It answers "is this residue a target that others look at?" A high value means many residues attend to it.

**Enrichment (Q1).**
How much more attention binding-site residues receive than an average residue.
Formula: (mean attention received by binding residues) / (mean attention received by all residues).
It is a unitless ratio, so it is comparable across proteins of different length. 1.5× means binding residues receive 50% more attention than a typical residue. 1.0× means no difference.

**Within-site attention (Q1).**
Of the attention that binding residues send out, the fraction that lands on *other* binding residues, compared to what you would expect by chance. The "× over random" baseline divides by the fraction of residues that are binding sites, so a value of 3× means binding residues attend to each other three times more than a random set of residues of the same size would. (This metric has a known confound - see the contiguity control in "What still needs checking".)

**Disorder preference (Q2.1).**
Like enrichment, but comparing disordered vs structured residues instead of binding vs non-binding.
Formula: (mean attention received by disordered residues) / (mean attention received by structured residues), computed per head.
Above 1 means the head attends more to disordered residues; below 1 means it prefers structured ones. Because both region types come from the *same* protein, protein-specific effects cancel out.

**Attention entropy (Q2.2).**
How focused or spread-out a residue's attention is. For residue `i` with attention row `A[i, :]`, entropy = `- sum over j of A[i, j] * ln(A[i, j])`, measured in **nats** (natural log). Low entropy = attention concentrated on a few residues (focused). High entropy = attention spread thinly over many residues (diffuse). The maximum possible value is `ln(L)`. We average entropy over disordered vs structured residues separately.

**Cross-boundary flow (Q2.3).**
At a disorder/structure boundary, the fraction of a residue's attention that crosses to the *other* side of the boundary. Because each attention row already sums to 1, summing a residue's attention to all residues on the far side gives a fraction directly (0 to 1). We compute this as a function of distance from the boundary (offset 0 = right at the edge). A peak at the boundary means edge residues split their attention across both regions.

**AUPRC and MCC (Q3).**
These score how well a simple probe recovers binding sites from a layer's embeddings. **AUPRC** (area under the precision-recall curve) is robust to class imbalance; its floor is the fraction of positive (binding) residues, so a type with 2% binding has a much lower AUPRC floor than one with 31%. **MCC** (Matthews correlation coefficient) is a balanced score from -1 to +1. We use these two as primary because AUROC is inflated when negatives dominate.

---

## Q1 - Attention at binding sites

*Research question: How does attention behave at known binding sites? Do binding site residues attend to each other?*

**How I did it:** For every protein, layer, and head, I computed two things from the saved attention tensors: **enrichment** (defined above) and **within-site attention** (defined above). I averaged across all proteins per binding type. Proteins where every residue is a binding site are excluded (enrichment is undefined there).

### Disordered

| Binding type | Enrichment (best layer) | Within-site attention (vs random) |
|---|---|---|
| Ion | 1.52× (layer 8) | 3.2× (layer 32) |
| Protein | 1.49× (layer 8) | 3.5× (layer 32) |
| DNA | 1.46× (layer 3) | 3.2× (layer 3) |
| RNA | 1.75× (layer 29) | 3.0× (layer 3) |

Binding site residues attract more attention than random (1.4-1.7×) - a modest but consistent effect - and they attend strongly to each other (~3× over random), peaking in the final layers.

### Structured vs disordered (enrichment best layer)

| Type | Disordered | Structured |
|---|---|---|
| Ion | 1.52× (layer 8) | 5.11× (layer 17) |
| Protein | 1.49× (layer 8) | 1.14× (layer 31) |
| DNA | 1.46× (layer 3) | 2.10× (layer 33) |
| RNA | 1.75× (layer 29) | 1.95× (layer 31) |

Structured binding sites generally show a stronger and later attention signal than disordered ones (ion especially, 5.11× vs 1.52×) - consistent with structured sites depending on the folded 3D context that ESM2 builds in deeper layers. Structured protein binding is the exception, with weak enrichment (1.14×), reflecting how diffuse protein-protein interfaces are.

![Figure: Q1 enrichment by layer, disordered vs structured](outputs/combined_figures/q1_disordered_vs_structured.png)

**Figure: Q1 enrichment by layer, disordered vs structured.** 


X-axis: ESM2 layer (1-33). Y-axis: attention enrichment (ratio; 1.0 = no difference from average, shown as a grey line). Each color is a binding type; solid lines are disordered proteins, dashed lines are structured. The curve shows, at each layer, the enrichment of the *best-performing head* at that layer (max across the 20 heads). The figure makes the depth difference visible: structured curves (dashed) rise higher and peak in deeper layers than the disordered curves (solid). *Caveat for the caption: because the curve takes the maximum over 20 heads, it is slightly optimistic (it always picks the luckiest head); a mean-across-heads band would show the typical head too.*

![Figure: Q1 enrichment heatmap](outputs/q1_figures/fig1_enrichment_heatmaps.png)

**Figure: Q1 enrichment heatmap**

X-axis: head (0-19). Y-axis: layer (1-33). Color: enrichment. A star marks the strongest cell. This shows the signal is concentrated in a few specific heads rather than spread across all of them, which supports the claim that specific heads specialize for binding.

**Caveat:** the within-site metric is partly inflated by binding sites being contiguous stretches combined with attention's local bias. The "× over random" baseline controls for the number of binding sites but not for contiguity - a follow-up control is described at the end of this report.

To put it simply: binding sites usually sit in continuous stretches along the sequence, and ESM2's attention includes positional heads that attend preferentially to nearby residues. So a binding residue attends to its neighbors - which happen to also be binding residues just because they're next to each other, not because the model specially links binding sites. The contiguity control (see the end of the report) measures how much of the signal survives once this is accounted for.

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

This is the clearest result. For **disordered** ion and DNA binding, information peaks in early layers (4-5); for **structured** binding of the same types, it peaks late (31). Disorder makes binding sites readable earlier in the network, because there is no structural context to integrate; structured sites require the full depth. The last layer is not universally best - only protein binding peaks at the final layer in the disordered set.

![Figure: Q3 AUPRC by layer, disordered vs structured](outputs/combined_figures/q3_disordered_vs_structured.png)

**Figure: Q3 AUPRC by layer, disordered vs structured.**

A 2×2 grid, one panel per binding type. X-axis: ESM2 layer. Y-axis: AUPRC (higher = binding sites more decodable from that layer's embeddings). Solid line = disordered, dashed = structured; faint vertical lines mark the best layer for each. The panels show disordered ion and DNA peaking early (left side of the axis) while their structured counterparts peak late (right side). *Recommended addition to the caption: a dashed horizontal line at each type's positive rate (the AUPRC floor), so the reader can see how far above chance each curve sits.*

![Figure: best layer vs last layer](outputs/q3_figures/fig3_best_layers.png)

**Figure (recommended to add): best layer vs last layer.**

Left panel: a bar per binding type showing its best layer, with a dashed line at layer 33 (the last layer). Right panel: best-layer AUPRC vs last-layer AUPRC side by side. This figure directly visualizes the practical claim "the last layer is not universally best," which otherwise appears only in the text.

**Note on absolute values:** structured AUPRC is lower than disordered (e.g. ion 0.42 vs 0.56), but this mostly reflects the much lower positive-class density in the structured data (ion ~2.3% binding vs ~31% in DisProt). The robust comparison is the *layer pattern* (where the peak is), not the absolute height. Because AUPRC's floor is the positive rate, the two datasets should not be compared on raw height - only on where the peak sits.

Q1 (attention) and Q3 (probing) measure different things - Q1 looks at each layer's attention, Q3 at each layer's embeddings - so their exact best layers are not expected to match, and Q1's two metrics themselves often peak at different depths. But the broad depth trend agrees: DNA binding peaks early in both (Q1 enrichment layer 3, Q3 layer 4), while protein binding peaks late in both (Q1 within-site layer 32, Q3 layer 33). That two independent methods point the same way strengthens the finding.

---

## Q2 - Attention in disordered vs structured regions

*Research question: Which layers attend to IDP regions? How does attention behave differently in disordered vs structured regions? How does it behave at the edge (boundary)?*

**How I did it:** Using the DisProt disorder annotations, I ran three analyses on the 720 mosaic proteins - proteins with reliable per-residue disorder labels that contain both disordered and structured regions - each comparing the two region types *within* the same protein (which controls for protein-specific effects). This within-protein design is why the separate structured dataset is not needed for Q2: both region types are already present, and labeled, inside these proteins. **Q2.1** measures attention received by disordered vs structured residues. **Q2.2** measures attention entropy. **Q2.3** measures cross-boundary flow. All three metrics are defined in the Metrics section above.

### Q2.1 - Which layers attend to disordered regions

| Type | Best layer | Disorder preference |
|---|---|---|
| Ion | 12 | 10.09× |
| Protein | 2 | 2.31× |
| DNA | 22 | 2.67× |
| RNA | 22 | 10.52× |

Specific heads attend up to 10× more to disordered residues than to structured ones. ESM2 clearly distinguishes disordered regions, though the depth at which it does so varies by binding type.

![Figure: Q2.1 disorder preference heatmap](outputs/q2_figures/fig1_disorder_preference.png)

**Figure: Q2.1 disorder preference heatmap.** 

A 2×2 grid, one panel per binding type. X-axis: head (0-19). Y-axis: layer (1-33). Color: disorder preference (above 1 = attends more to disordered residues; a star marks the strongest head). The value reported in the table is the single strongest cell. *Note for the caption: because this is a ratio metric, the color scale should be centered at 1 (a log or diverging scale), so that "2× toward disordered" and "0.5× toward structured" appear equally far from the center.* A companion line figure (`fig4_preference_by_layer.png`) shows the max preference per layer, making the depth at which each type peaks easy to read.

### Q2.2 - How attention behaves (entropy)

| Type | Disordered entropy | Structured entropy | Difference |
|---|---|---|---|
| Ion | 3.846 | 3.794 | +0.052 |
| Protein | 4.062 | 3.922 | +0.140 |
| DNA | 3.993 | 3.873 | +0.120 |
| RNA | 4.161 | 4.044 | +0.117 |

Disordered residues have consistently more diffuse attention (higher entropy, in nats) than structured residues, in all four binding types. The effect is modest but consistent in direction - structured residues attend in a more focused way (they have specific structural contacts), while disordered residues spread their attention more (they are flexible, with no fixed partners).

![Figure: Q2.2 entropy comparison](outputs/q2_figures/fig2_entropy.png)

**Figure: Q2.2 entropy comparison.** 

X-axis: binding type. Y-axis: mean attention entropy in nats (higher = more diffuse). Two bars per type: disordered (solid) and structured (faded). In every type the disordered bar is higher. **Important honesty note for the caption:** the y-axis in the current figure is zoomed (it does not start at 0), which visually magnifies a small difference (+0.05 to +0.14 nats). The caption must state that the axis is truncated, and ideally annotate the actual difference and its significance on each pair. The differences are small in absolute terms; their value is that the direction is identical across all four types.

### Q2.3 - Behavior at the edge (boundary)

| Type | At boundary (offset 0) | Far from boundary (±20) |
|---|---|---|
| Ion | 0.465 | ~0.26 |
| Protein | 0.480 | ~0.27 |
| DNA | 0.487 | ~0.27 |
| RNA | 0.489 | ~0.28 |

Cross-boundary attention peaks sharply at the boundary and drops off into either region - again identical across all four types. Residues right at the disorder/structure edge send ~48% of their attention across to the other side; residues deep within a region send only ~27%. Boundary residues integrate information from both sides, which is exactly the "edge" behavior the research question asks about.

![Figure: Q2.3 boundary profile](outputs/q2_figures/fig3_boundary_profile.png)

**Figure: Q2.3 boundary profile** 

X-axis: distance from the disorder/structure boundary in residues (negative = one side, 0 = the boundary, positive = the other side). Y-axis: fraction of attention that crosses the boundary (0 to 1). A dashed vertical line marks the boundary. Each color is a binding type. All four curves peak at offset 0 and fall away symmetrically, showing that edge residues split their attention across both regions while interior residues stay local. *Note: a residue near a boundary is, by simple sequence proximity, close to residues of the other type, so part of this peak could come from the same local-attention effect discussed in Q1. Worth one sentence acknowledging it.*

---

## What I take from this

The central theme, consistent across both Q1 and Q3, is that **binding information lives at different network depths depending on both the binding type and the structural context.** Disordered ion and DNA binding, which rely on local sequence chemistry, are captured early; structured binding of the same types needs the full depth to integrate 3D context. Two independent methods - attention (Q1) and probing (Q3) - agree on this, which strengthens it.

Q2 adds a third, consistent picture: ESM2 not only distinguishes disordered from structured residues (up to 10× attention preference, more diffuse attention on disordered regions), it also treats the boundary between them as special, concentrating cross-region attention right at the edge. All three Q2 metrics give the same answer across all four binding types.

One practical implication runs through the whole analysis: the research project used last-layer embeddings throughout, but these results suggest the optimal layer depends on binding type and structural context - for disordered ion/DNA, layers 4-5 clearly beat the last layer.

---

## Methodological notes

- AUROC is inflated on imbalanced binding data, so AUPRC and MCC are the primary metrics; all three agree on the best layer for every type.
- Q2 uses real DisProt disorder annotations, not a predictor - DisProt curates disorder regions directly.
- For the boundary window, rather than picking an arbitrary size I analyze attention across the full profile (−20 to +20) and read the transition width from it; summary stats use ±5 (the scale of disordered binding motifs).
- Q2 within-protein comparison (disordered vs structured residues in the same chain) controls for protein-level confounds.
- Proteins longer than 1,800 residues were excluded from Q2 (40 of 720, ~5.6%) due to the memory cost of the entropy computation, leaving 680 proteins used in the Q2 tables; length is unrelated to the disorder question, so this is not expected to bias the result (worth a quick check that disorder fraction does not correlate with length in the excluded set).

---

## What still needs checking (before final write-up)

The results above are point estimates - single averaged numbers. Two additions turn "modest but consistent" from an assertion into a demonstrated result. Both read from the already-saved per-protein data, so neither requires re-running ESM2.

### 1. Statistical significance and error bars

Right now every table gives one number with no measure of spread, so a reader cannot tell whether a small difference (for example the +0.05 nat entropy gap in Q2.2) is real or noise. Two simple additions fix this.

**Paired test for Q2.1 and Q2.2.** For each protein we have a disordered value and a structured value from the *same* protein, so the two are paired. Collect the per-protein difference (disordered − structured) across all proteins and run a **Wilcoxon signed-rank test** (the paired, non-parametric test; safer than a t-test because these values are not bell-shaped). This gives a p-value for "is the difference reliably different from zero". Alongside the p-value, report an **effect size** (for example Cohen's dz, the mean difference divided by the spread of the differences), because with ~680 proteins even a tiny difference will be "significant" - the effect size tells you whether it is also *meaningful*. The likely honest outcome for Q2.2 is "significant but small", which is a perfectly good result.

**Bootstrap confidence bands for Q1 and Q3 curves.** The layer curves are averages over proteins. To show how stable each curve is, resample the proteins with replacement (draw a random set of the same size, allowing repeats), recompute the per-layer average, and repeat about 1,000 times. At each layer, the 2.5th and 97.5th percentiles of those repeats give a shaded band around the line. If the early-layer band and the late-layer band do not overlap, the "disordered peaks early, structured peaks late" claim is solid; if they overlap, the claim must be softened. For Q3 this also means "best layer" should be read as a small *region* of layers whose bands overlap the peak, not a single exact layer (layer 4 vs 5 is almost certainly within noise).

### 2. Contiguity control for the Q1 within-site metric

**The problem in plain terms.** The within-site metric says binding residues attend to each other ~3× more than chance, and we would like to read that as "the model links binding sites together". But there is an innocent alternative. Binding sites come in *contiguous runs* (a motif is several residues in a row), and ESM2's attention includes a local component that attends to nearby residues. So a binding residue attends to its immediate neighbors - which are also binding residues simply because they sit next to each other. The 3× could be geometry (local attention landing on a contiguous block), not binding-specific linking.

**The control.** For each protein, keep the *shape* of the real binding sites (the same number of residues, in contiguous runs of the same lengths) but move those runs to random positions elsewhere in the sequence that are not real binding sites. Then compute the exact same within-site metric on these fake blocks. Because the fake blocks are also contiguous, they get the same benefit from local attention. Compare:

- If the fake blocks also show ~3×, the effect is geometry, and the binding-specific claim does not hold.
- If the fake blocks show, say, ~1.8× and real binding shows ~3×, then ~1.8× is the geometry contribution and the extra ~1.2× is the genuine binding-specific signal - and that difference is what can be claimed.

Implementation is light: read the real binding mask, measure the run-lengths of its contiguous blocks, drop blocks of those same lengths at random non-overlapping positions, run the existing within-site function on that fake mask, repeat ~50-100 times per protein, and average to get a stable per-protein baseline. Report real vs matched-baseline side by side.

The same local-attention logic mildly applies to the Q2.3 boundary peak (residues near a boundary are near the other region by proximity alone); one sentence acknowledging this, and ideally a check that the peak exceeds a plain distance-decay baseline, would close that loophole too.

### 3. Other checks

- **Probe split.** State explicitly whether the Q3 train/test split is clustered by sequence identity. If train and test share close homologs, probe scores are inflated; since the pipeline already uses mmseqs2 clustering elsewhere, the probe split should be identity-controlled too.
- **Single model.** Everything uses one ESM2 checkpoint and one run. At minimum note this as a limitation; if feasible, replicating one key result (e.g. the Q3 layer pattern) on a smaller ESM2 would show it is a property of the model family, not this checkpoint.

---

## Next steps

- Add the paired tests, effect sizes, and bootstrap bands described above, and regenerate the Q1/Q3 figures with the shaded bands and the Q2.2 axis fix.
- Run the contiguity control and report real vs matched-baseline within-site values.
- (Optional) Compare structured regions in IDPs against structured regions in fully structured proteins, to check whether structural context differs between the two.
- Q1 second part: compare correctly-predicted vs missed binding sites (needs predictions from the research-project model).
- Integrate all three questions into the thesis write-up.
- Continue the literature review on transformer interpretability.