"""
ESM2 Inference Pipeline — Full Dataset
=======================================
Processes proteins from CSV or TSV files, extracting attention tensors
and layer embeddings from ESM2.

Handles BOTH data formats automatically:
  DisProt format:    columns protein_id, sequence, labels (comma-separated "0,1,1,0")
                     tab-separated (.tsv)
  Structured format: columns pdb_id, chain_id, sequence, annotation (no commas "0110")
                     comma-separated (.csv)

Key features:
  - Loads ESM2 once, loops over all proteins
  - Skips already-processed proteins (safe to resubmit if job fails)
  - Handles OOM for long proteins (embeddings only if L > max_length)
  - Per-protein error handling

Outputs per protein (.npz):
  - embeddings: [34 x L x 1280]   float16
  - attentions: [33 x 20 x L x L] float16  (skipped if L > max_length)
  - labels:     [L]                int8

Usage:
  python pipeline.py --input file1.csv file2.tsv --output outputs/
"""

import os
import argparse
import torch
import esm
import numpy as np
import pandas as pd

DEFAULT_MAX_LENGTH = 3000


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
def load_model(device):
    print("Loading ESM2 model...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model = model.to(device)
    print(f"Model loaded on {device}\n")
    return model, alphabet, batch_converter


# ─────────────────────────────────────────────
# FORMAT-AWARE FILE LOADING
# ─────────────────────────────────────────────
def load_file(path):
    """
    Load a CSV or TSV file and return a dataframe plus a parser function
    that extracts (protein_id, sequence, labels) from each row.

    Auto-detects separator and column format.
    """
    # Detect separator: .tsv → tab, .csv → comma
    sep = '\t' if path.endswith('.tsv') else ','
    df = pd.read_csv(path, sep=sep)
    cols = set(df.columns)

    # Detect format from columns
    if "protein_id" in cols and "labels" in cols:
        # DisProt format: protein_id + comma-separated labels
        fmt = "disprot"
    elif "pdb_id" in cols and "annotation" in cols:
        # Structured format: pdb_id + chain_id + no-comma annotation
        fmt = "structured"
    else:
        raise ValueError(f"Unrecognized columns in {path}: {sorted(cols)}")

    return df, fmt


def parse_row(row, fmt):
    """Extract (protein_id, sequence, labels) based on detected format."""
    sequence = str(row['sequence'])

    if fmt == "disprot":
        protein_id = str(row['protein_id'])
        labels = np.array([int(x) for x in row['labels'].split(',')], dtype=np.int8)
    else:  # structured
        protein_id = f"{row['pdb_id']}_{row['chain_id']}"
        labels = np.array([int(c) for c in str(row['annotation'])], dtype=np.int8)

    return protein_id, sequence, labels


# ─────────────────────────────────────────────
# SINGLE PROTEIN PROCESSING
# ─────────────────────────────────────────────
def process_protein(protein_id, sequence, labels, model,
                    batch_converter, device, output_dir, max_length):
    """Run ESM2 on one protein and save outputs."""
    save_path = os.path.join(output_dir, f"{protein_id}.npz")

    if os.path.exists(save_path):
        print(f"  SKIP     {protein_id:<18} (already exists)")
        return "skipped"

    L = len(sequence)

    # Verify labels align with sequence
    if len(labels) != L:
        print(f"  MISMATCH {protein_id:<18} seq L={L} but labels={len(labels)} — skipping")
        return "error"

    too_long = L > max_length
    if too_long:
        print(f"  LONG     {protein_id:<18} L={L} > {max_length} — embeddings only")

    try:
        data = [("protein", sequence)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        with torch.no_grad():
            results = model(
                tokens,
                repr_layers=list(range(34)),
                need_head_weights=not too_long
            )

        embeddings = torch.stack([
            results["representations"][layer][0, 1:-1, :]
            for layer in range(34)
        ]).cpu()

        if not too_long:
            attentions = results["attentions"][0, :, :, 1:-1, 1:-1].cpu()
        else:
            attentions = None

        assert embeddings.shape == (34, L, 1280), f"Embedding shape error: {embeddings.shape}"
        if attentions is not None:
            assert attentions.shape == (33, 20, L, L), f"Attention shape error: {attentions.shape}"

        os.makedirs(output_dir, exist_ok=True)
        save_dict = {
            "embeddings": embeddings.numpy().astype(np.float16),
            "labels":     labels
        }
        if attentions is not None:
            save_dict["attentions"] = attentions.numpy().astype(np.float16)

        np.savez_compressed(save_path, **save_dict)

        size_mb = os.path.getsize(save_path) / (1024 ** 2)
        tag = "full" if attentions is not None else "emb_only"
        print(f"  OK       {protein_id:<18} L={L:<6} {size_mb:>7.1f} MB  [{tag}]")
        return "ok" if not too_long else "ok_emb_only"

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  OOM      {protein_id:<18} L={L} — skipping")
            torch.cuda.empty_cache()
            return "oom"
        print(f"  ERROR    {protein_id:<18} {str(e)[:50]}")
        return "error"
    except Exception as e:
        print(f"  ERROR    {protein_id:<18} {str(e)[:50]}")
        return "error"


# ─────────────────────────────────────────────
# FILE PROCESSING
# ─────────────────────────────────────────────
def process_file(path, output_dir, model, batch_converter, device, max_length):
    df, fmt = load_file(path)
    filename = os.path.basename(path)
    print(f"\n{'='*60}")
    print(f"  File: {filename}  ({len(df)} proteins, format={fmt})")
    print(f"{'='*60}")

    counts = {"ok": 0, "ok_emb_only": 0, "skipped": 0, "oom": 0, "error": 0}

    for _, row in df.iterrows():
        protein_id, sequence, labels = parse_row(row, fmt)
        result = process_protein(
            protein_id, sequence, labels,
            model, batch_converter, device, output_dir, max_length
        )
        counts[result] += 1

    print(f"\n  File summary: {counts}")
    return counts


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main(files, output_dir, max_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:     {device}")
    print(f"Output dir: {output_dir}")
    print(f"Max length: {max_length} (attention skipped above this)")

    model, alphabet, batch_converter = load_model(device)

    total = {"ok": 0, "ok_emb_only": 0, "skipped": 0, "oom": 0, "error": 0}

    for path in files:
        if not os.path.exists(path):
            print(f"\nWARNING: {path} not found — skipping")
            continue
        counts = process_file(path, output_dir, model, batch_converter, device, max_length)
        for k in total:
            total[k] += counts[k]

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Processed (full):      {total['ok']}")
    print(f"  Processed (emb only):  {total['ok_emb_only']}")
    print(f"  Skipped (exists):      {total['skipped']}")
    print(f"  OOM errors:            {total['oom']}")
    print(f"  Other errors:          {total['error']}")
    print(f"  Total:                 {sum(total.values())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESM2 inference pipeline — handles CSV and TSV")
    parser.add_argument('--input', required=True, nargs='+', help='One or more CSV/TSV files')
    parser.add_argument('--output', required=True, help='Output directory for .npz files')
    parser.add_argument('--max_length', type=int, default=DEFAULT_MAX_LENGTH,
                        help=f'Skip attention above this length (default: {DEFAULT_MAX_LENGTH})')
    args = parser.parse_args()

    main(args.input, args.output, args.max_length)