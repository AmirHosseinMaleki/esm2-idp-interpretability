import time
import torch
import esm
import numpy as np

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB\n")

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()
model = model.to(device)

# Test specific lengths that cover our dataset range
test_lengths = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000]

print(f"{'L':>6}  {'Time (s)':>10}  {'Peak mem (GB)':>14}  {'Estimated for 992 proteins':>28}")
print("-" * 70)

times = []
for L in test_lengths:
    # Create a fake sequence of length L
    sequence = "A" * L
    data = [("protein", sequence)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()   # wait for GPU to be ready
    start = time.time()

    with torch.no_grad():
        results = model(
            tokens,
            repr_layers=list(range(34)),
            need_head_weights=True
        )

    torch.cuda.synchronize()   # wait for GPU to finish
    elapsed = time.time() - start
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    times.append((L, elapsed))
    print(f"{L:>6}  {elapsed:>10.3f}  {peak_mem:>14.2f}")

# Fit quadratic (time ~ L^2) and estimate total
L_arr = np.array([t[0] for t in times])
T_arr = np.array([t[1] for t in times])
coeffs = np.polyfit(L_arr, T_arr, 2)

# Load actual dataset lengths
import pandas as pd
import os
base = '/work/malekia/esm2-idp-interpretability/data/csv_prepared_data/'
all_lengths = []
for f in os.listdir(base):
    if f.endswith('.tsv'):
        df = pd.read_csv(base + f, sep='\t')
        all_lengths.extend(df['length'].tolist())

estimated_total = sum(
    np.polyval(coeffs, min(l, 3000))   # cap at 3000 (our threshold)
    for l in all_lengths
)

print(f"\nEstimated total runtime: {estimated_total/3600:.1f} hours")
print(f"(capped at L=3000 for long proteins, {len(all_lengths)} proteins total)")
