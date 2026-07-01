"""
Fetch DisProt Disorder Annotations
===================================
For each protein in the DisProt binding files, fetch its disorder
regions from the DisProt API and build a per-residue disorder label.

Output: data/disorder_annotations/<binding_type>_disorder.tsv
  Same format as the binding files, plus:
    - disorder_labels    : "0,1,1,0,..." (1 = disordered residue)
    - disorder_coverage  : % of residues that are disordered

Disorder definition:
  A residue is disordered (1) if covered by a DisProt region with
  term_namespace == "Structural state" and term_name == "disorder"
  (IDPO:0000002). Everything else is structured (0).

Usage:
  python fetch_disorder.py
"""

import os
import time
import json
import urllib.request
import numpy as np
import pandas as pd

DATA_DIR = "/work/malekia/esm2-idp-interpretability/data/csv_prepared_data"
OUT_DIR  = "/work/malekia/esm2-idp-interpretability/data/disorder_annotations"
API_URL  = "https://disprot.org/api/{}"

# Source binding files (one set per binding type)
BINDING_FILES = {
    "ion":     ["ion_binding_train.tsv",     "ion_binding_val.tsv",     "ion_binding_test.tsv"],
    "protein": ["protein_binding_train.tsv", "protein_binding_val.tsv", "protein_binding_test.tsv"],
    "dna":     ["dna_binding_train.tsv",     "dna_binding_val.tsv",     "dna_binding_test.tsv"],
    "rna":     ["rna_binding_train.tsv",     "rna_binding_val.tsv",     "rna_binding_test.tsv"],
}

REQUEST_DELAY = 0.3   # seconds between API calls (be polite to the server)


def fetch_disprot(protein_id):
    """
    Fetch a protein's DisProt entry. Returns the parsed JSON, or None.
    Strips isoform suffix (P12345-1 -> P12345) since DisProt uses
    canonical accessions.
    """
    # DisProt maps to canonical UniProt — strip isoform suffix
    clean_id = protein_id.split('-')[0]
    url = API_URL.format(clean_id)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return None


def build_disorder_labels(entry, seq_length):
    """
    From a DisProt entry, build a per-residue disorder array [L].
    1 = disordered, 0 = structured.

    Uses regions where term_namespace == "Structural state"
    and term_name == "disorder".
    """
    labels = np.zeros(seq_length, dtype=np.int8)

    for region in entry.get("regions", []):
        ns   = region.get("term_namespace", "")
        name = region.get("term_name", "")
        if ns == "Structural state" and name == "disorder":
            # DisProt positions are 1-based inclusive
            start = region["start"] - 1      # to 0-based
            end   = region["end"]            # exclusive upper in slice
            start = max(0, start)
            end   = min(seq_length, end)
            labels[start:end] = 1

    return labels


def process_binding_type(binding_type, files):
    print(f"\n{'='*60}")
    print(f"  {binding_type.upper()}")
    print(f"{'='*60}")

    # Combine all splits, dedup by protein
    dfs = []
    for f in files:
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            dfs.append(pd.read_csv(path, sep='\t'))
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset='protein_id')
    print(f"  Proteins to fetch: {len(df)}")

    rows = []
    n_found = 0
    n_missing = 0
    n_seq_mismatch = 0

    for i, row in df.iterrows():
        protein_id = str(row['protein_id'])
        sequence   = str(row['sequence'])
        L          = len(sequence)

        entry = fetch_disprot(protein_id)
        time.sleep(REQUEST_DELAY)

        if entry is None:
            n_missing += 1
            continue

        # Check sequence matches (disorder positions must align)
        disprot_seq = entry.get("sequence", "")
        if disprot_seq != sequence:
            # Length mismatch means positions won't align — skip
            if len(disprot_seq) != L:
                n_seq_mismatch += 1
                continue

        disorder = build_disorder_labels(entry, L)
        disorder_coverage = round(100 * disorder.sum() / L, 2)

        # Build output row: keep original columns + new disorder fields
        out = row.to_dict()
        out['disorder_labels']   = ','.join(map(str, disorder.tolist()))
        out['disorder_coverage'] = disorder_coverage
        rows.append(out)
        n_found += 1

        if (i + 1) % 50 == 0:
            print(f"    progress: {i+1}/{len(df)}  (found {n_found})")

    print(f"\n  Found:          {n_found}")
    print(f"  Missing (API):  {n_missing}")
    print(f"  Seq mismatch:   {n_seq_mismatch}")

    if rows:
        out_df = pd.DataFrame(rows)
        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(OUT_DIR, f"{binding_type}_disorder.tsv")
        out_df.to_csv(out_path, sep='\t', index=False)

        avg_cov = out_df['disorder_coverage'].mean()
        fully   = (out_df['disorder_coverage'] >= 95).sum()
        partial = ((out_df['disorder_coverage'] > 5) &
                   (out_df['disorder_coverage'] < 95)).sum()
        print(f"  Saved: {out_path}")
        print(f"  Avg disorder coverage: {avg_cov:.1f}%")
        print(f"  Fully disordered (>=95%): {fully}")
        print(f"  Partially disordered (5-95%): {partial}  <- usable for boundary analysis")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for binding_type, files in BINDING_FILES.items():
        process_binding_type(binding_type, files)
    print(f"\n✓ Disorder annotations saved to {OUT_DIR}")


if __name__ == "__main__":
    main()