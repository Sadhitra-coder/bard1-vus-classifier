import pandas as pd
import pyBigWig
import numpy as np

import os

# Get the workspace root
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Path to the bigWig file
bw_path = os.path.join(WORKSPACE_ROOT, "data", "raw", "hg38.phyloP100way.bw")

# Load the original variant file (has CHROM, POS, REF, ALT, and also label? we need coordinates)
# We'll use bard1_variants.csv (which includes CHROM, POS)
variants_file = os.path.join(WORKSPACE_ROOT, "data", "intermediate", "bard1_variants.csv")
variants = pd.read_csv(variants_file)
print(f"Loaded {len(variants)} variants from {variants_file}")

# Ensure CHROM is in 'chr2' format (without 'chr'? bigWig expects 'chr2')
variants['CHROM'] = variants['CHROM'].astype(str)
if not variants['CHROM'].str.startswith('chr').any():
    variants['CHROM'] = 'chr' + variants['CHROM']
print(variants['CHROM'].head())

# Open the bigWig file
bw = pyBigWig.open(bw_path)

# Extract phyloP score for each position (1‑based POS → 0‑based start)
def get_phylop(row):
    chrom = row['CHROM']
    pos = int(row['POS'])
    try:
        # bigWig values are for intervals: start (0‑based), end (1‑based)
        val = bw.values(chrom, pos-1, pos)[0]
        return val if val is not None else 0.0
    except:
        return 0.0

variants['phyloP'] = variants.apply(get_phylop, axis=1)
bw.close()

print("PhyloP scores extracted. Summary:")
print(variants['phyloP'].describe())

# Save only the coordinates and phyloP score (for merging)
phylop_df = variants[['CHROM', 'POS', 'phyloP']].copy()
phylop_output = os.path.join(WORKSPACE_ROOT, "data", "intermediate", "phylop_scores.csv")
phylop_df.to_csv(phylop_output, index=False)
print(f"Saved phyloP scores to {phylop_output}")


feat = pd.read_csv(os.path.join(WORKSPACE_ROOT, "data", "processed", "bard1_features_with_af.csv"))
phylop = pd.read_csv(os.path.join(WORKSPACE_ROOT, "data", "intermediate", "phylop_scores.csv"))
feat['phyloP'] = phylop['phyloP']  # assuming same row order
output_path = os.path.join(WORKSPACE_ROOT, "data", "processed", "bard1_features_with_phylop.csv")
feat.to_csv(output_path, index=False)
print(f"Saved features with phyloP to {output_path}")