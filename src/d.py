import pandas as pd

df = pd.read_csv('/work/malekia/esm2-idp-interpretability/data/csv_prepared_data/dna_binding_train.tsv', sep='\t')
row = df[df['protein_id'] == 'P17096'].iloc[0]

print('Protein ID:      ', row['protein_id'])
print('Length:          ', row['length'])
print('Binding sites:   ', row['binding_positions'])
print('Coverage:        ', row['coverage_percent'], '%')
print('Annotations:     ', row['annotations_count'])
print('GO terms:        ', row['go_terms'])
print('Header:          ', row['header'])
print()
print('First 20 labels: ', row['labels'][:40])
print('Last 20 labels:  ', row['labels'][-40:])