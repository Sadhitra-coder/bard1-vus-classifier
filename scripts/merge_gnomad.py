import pandas as pd
import numpy as np

# 1. Load original variant file (has coordinates and label)
original = pd.read_csv("bard1_variants.csv")
print("Original columns:", original.columns.tolist())
# Keep only what we need for merging and label
original = original[['CHROM', 'POS', 'REF', 'ALT', 'label']].copy()

# 2. Load gnomAD results
gnomad = pd.read_csv("gnomad_frequencies.csv")
print("gnomAD columns:", gnomad.columns.tolist())

# Split variant_id into CHROM, POS, REF, ALT
def split_variant_id(vid):
    parts = str(vid).split('-')
    if len(parts) >= 4:
        chrom = parts[0]
        pos = parts[1]
        ref = parts[2]
        alt = '-'.join(parts[3:])   # ALT may contain '-'
        return chrom, pos, ref, alt
    return None, None, None, None

gnomad[['CHROM', 'POS', 'REF', 'ALT']] = gnomad['variant_id'].apply(
    lambda x: pd.Series(split_variant_id(x))
)
gnomad['POS'] = pd.to_numeric(gnomad['POS'], errors='coerce')
gnomad_clean = gnomad[['CHROM', 'POS', 'REF', 'ALT', 'joint_af']].dropna(subset=['POS'])

# 3. Convert types to ensure merge works
original['CHROM'] = original['CHROM'].astype(str)
gnomad_clean['CHROM'] = gnomad_clean['CHROM'].astype(str)
original['POS'] = original['POS'].astype(int)
gnomad_clean['POS'] = gnomad_clean['POS'].astype(int)

# 4. Merge gnomAD frequencies into original
original_with_af = original.merge(
    gnomad_clean,
    on=['CHROM', 'POS', 'REF', 'ALT'],
    how='left'
)

# Fill missing AF with a small value (variant not found in gnomAD = very rare)
original_with_af['joint_af'] = original_with_af['joint_af'].fillna(1e-6)
original_with_af['log_af'] = np.log10(original_with_af['joint_af'])

# 5. Load cleaned features (without coordinates)
features = pd.read_csv("bard1_features_cleaned.csv")
features = features.reset_index(drop=True)

# 6. Add AF columns (assuming same row order; if not, merge on index)
if len(features) == len(original_with_af):
    features['joint_af'] = original_with_af['joint_af']
    features['log_af'] = original_with_af['log_af']
else:
    # Fallback: merge using row index (if order is same)
    features = features.reset_index(drop=True)
    original_with_af = original_with_af.reset_index(drop=True)
    features['joint_af'] = original_with_af['joint_af']
    features['log_af'] = original_with_af['log_af']

# Ensure label is present (features already has label from previous step)
if 'label' not in features.columns:
    features['label'] = original_with_af['label']

# 7. Save final feature set
features.to_csv("bard1_features_with_af.csv", index=False)
print("Shape:", features.shape)
print("New columns added: joint_af, log_af")
print(features[['joint_af', 'log_af']].head())