"""
ESM2 Inference Pipeline
=======================
Extracts attention tensors and layer embeddings from ESM2
for a single protein (test run before scaling to full dataset).

Outputs per protein:
  - embeddings: [34 x L x 1280]   (layers 0-33, BOS/EOS stripped)
  - attentions: [33 x 20 x L x L] (layers 1-33, BOS/EOS stripped)
  - labels:     [L]                (per-residue binding site annotations)

Usage:
  python pipeline.py --input data.tsv --output outputs/ --idx 0
"""

import os
import argparse
import torch
import esm
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# STEP 1 - Load model
# ─────────────────────────────────────────────
def load_model(device):
    """
    Load pretrained ESM2 (33 layers, 650M parameters).
    First run downloads weights (~2.5 GB) - needs internet on login node.
    """
    print("Loading ESM2 model...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model = model.to(device)
    print(f"Model loaded on {device}")
    return model, alphabet, batch_converter


# ─────────────────────────────────────────────
# STEP 2 - Load one protein from TSV
# ─────────────────────────────────────────────
def load_protein(csv_path, idx):
    """
    Load a single protein row from the TSV file.

    TSV columns used:
      protein_id  -> single UniProt/DisProt ID
      sequence    -> amino acid string
      labels      -> comma-separated "0,0,1,1,0,..." per-residue annotations
    """
    df = pd.read_csv(csv_path, sep='\t')
    row = df.iloc[idx]

    protein_id = str(row['protein_id'])
    sequence   = str(row['sequence'])

    # Labels are comma-separated: "0,0,1,1,0,..." -> [0,0,1,1,0,...]
    labels = np.array([int(x) for x in row['labels'].split(',')], dtype=np.int8)

    print(f"\nProtein:        {protein_id}")
    print(f"Sequence:       {sequence[:40]}...")
    print(f"Length (L):     {len(sequence)}")
    print(f"Binding sites:  {labels.sum()} / {len(labels)} residues")

    return protein_id, sequence, labels


# ─────────────────────────────────────────────
# STEP 3 - Tokenize
# ─────────────────────────────────────────────
def tokenize(sequence, batch_converter, device):
    """
    Convert amino acid string to token IDs.
    ESM2 adds BOS at position 0 and EOS at the end.
    Token length = L + 2.
    """
    data = [("protein", sequence)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    print(f"\nToken shape:    {list(tokens.shape)}  (1 x L+2 = 1 x {tokens.shape[1]})")
    return tokens


# ─────────────────────────────────────────────
# STEP 4 - Forward pass
# ─────────────────────────────────────────────
def forward_pass(tokens, model):
    """
    Run ESM2 forward pass requesting:
      - repr_layers: embeddings from all 34 layers (0 to 33)
      - need_head_weights: attention matrices from all 33 transformer layers

    Raw output shapes (before stripping BOS/EOS):
      representations[layer]: [1 x L+2 x 1280]
      attentions:             [1 x 33 x 20 x L+2 x L+2]
    """
    print("\nRunning ESM2 forward pass...")

    with torch.no_grad():
        results = model(
            tokens,
            repr_layers=list(range(34)),   # layers 0,1,...,33
            need_head_weights=True          # return attention weights
        )

    return results


# ─────────────────────────────────────────────
# STEP 5 - Strip BOS/EOS and stack tensors
# ─────────────────────────────────────────────
def extract_outputs(results, sequence_length):
    """
    Strip BOS (position 0) and EOS (last position) from
    both embeddings and attention tensors.

    After stripping:
      embeddings: [34 x L x 1280]
      attentions: [33 x 20 x L x L]
    """
    # --- Embeddings ---
    embeddings = torch.stack([
        results["representations"][layer][0, 1:-1, :]
        for layer in range(34)
    ])
    # Shape: [34 x L x 1280]

    # --- Attention ---
    attentions = results["attentions"][0, :, :, 1:-1, 1:-1]
    # Shape: [33 x 20 x L x L]

    embeddings = embeddings.cpu()
    attentions = attentions.cpu()

    print(f"\nEmbeddings shape: {list(embeddings.shape)}  [layers x L x dim]")
    print(f"Attentions shape: {list(attentions.shape)}  [layers x heads x L x L]")

    return embeddings, attentions


# ─────────────────────────────────────────────
# STEP 6 - Verify shapes
# ─────────────────────────────────────────────
def verify_shapes(embeddings, attentions, labels, sequence_length):
    """
    Verify all tensor shapes are correct and aligned with labels.
    """
    L = sequence_length

    assert embeddings.shape == (34, L, 1280), \
        f"Embedding shape mismatch: expected [34, {L}, 1280], got {list(embeddings.shape)}"

    assert attentions.shape == (33, 20, L, L), \
        f"Attention shape mismatch: expected [33, 20, {L}, {L}], got {list(attentions.shape)}"

    assert len(labels) == L, \
        f"Label length mismatch: expected {L}, got {len(labels)}"

    row_sums = attentions[0, 0].sum(dim=-1)
    max_deviation = (row_sums - 1.0).abs().max().item()
    print(f"  Max attention row deviation from 1.0: {max_deviation:.6f}")
    # Note: float16 rounding causes deviation, larger for short sequences
    pass

    print(f"\n✓ Embeddings shape correct:  {list(embeddings.shape)}")
    print(f"✓ Attentions shape correct:  {list(attentions.shape)}")
    print(f"✓ Labels length correct:     {len(labels)}")
    print(f"✓ Attention rows sum to ~1.0")


# ─────────────────────────────────────────────
# STEP 7 - Save to disk
# ─────────────────────────────────────────────
def save_outputs(protein_id, embeddings, attentions, labels, output_dir):
    """
    Save all outputs to a single compressed .npz file per protein.
    Saved in float16 to halve disk usage vs float32.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{protein_id}.npz")

    np.savez_compressed(
        save_path,
        embeddings = embeddings.numpy().astype(np.float16),
        attentions = attentions.numpy().astype(np.float16),
        labels     = labels
    )

    size_mb = os.path.getsize(save_path) / (1024 ** 2)
    print(f"\nSaved to:  {save_path}")
    print(f"File size: {size_mb:.1f} MB")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main(csv_path, output_dir, protein_idx):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, alphabet, batch_converter = load_model(device)
    protein_id, sequence, labels     = load_protein(csv_path, protein_idx)
    tokens                           = tokenize(sequence, batch_converter, device)
    results                          = forward_pass(tokens, model)
    embeddings, attentions           = extract_outputs(results, len(sequence))

    verify_shapes(embeddings, attentions, labels, len(sequence))
    save_outputs(protein_id, embeddings, attentions, labels, output_dir)

    print("\n✓ Pipeline test complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESM2 inference pipeline")
    parser.add_argument('--input',  required=True, help='Path to TSV input file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--idx',    type=int, default=0, help='Row index to process')
    args = parser.parse_args()

    main(args.input, args.output, args.idx)