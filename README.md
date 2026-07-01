# ESM2-IDP-Interpretability

Analyzing ESM2 attention layers for intrinsically disordered protein (IDP) binding sites.

## Phase 0 — Inference pipeline

One ESM2 forward pass per protein; saves attention tensors and intermediate-layer
embeddings to disk so downstream experiments read from files instead of re-running ESM2.


## Roadmap
- Phase 1 (Q1): attention at binding sites
- Phase 2 (Q2): layer-wise attention to IDP regions & boundaries
- Phase 3 (Q3): probing binding-site info across layers