import pandas as pd
import numpy as np
import os

# Get the workspace root
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ============================================================
# 1. Load the TSV file (from your filtered output)
# ============================================================
tsv_file = os.path.join(WORKSPACE_ROOT, "data", "intermediate", "bard1_small.tsv")
df = pd.read_csv(tsv_file, sep='\t', low_memory=False)
print(f"Loaded {len(df)} rows")

# ============================================================
# 2. Select only the columns we need (map to expected names)
# ============================================================
# Expected columns in feature_engineering.py:
# CHROM, POS, REF, ALT, CLNSIG, CLNREVSTAT, MC, CLNDN, CLNVC, ORIGIN, RS, etc.

# Map from TSV column names to our standard names
rename_map = {
    'Chromosome': 'CHROM',
    'Start': 'POS',
    'ReferenceAlleleVCF': 'REF',
    'AlternateAlleleVCF': 'ALT',
    'ClinicalSignificance': 'CLNSIG',
    'ReviewStatus': 'CLNREVSTAT',
    'Type': 'MC',
    'VariantType': 'CLNVC',
    'OriginSimple': 'ORIGIN',
    'RS# (dbSNP)': 'RS',
    'GeneSymbol': 'GENEINFO',
    'PhenotypeList': 'CLNDN',
    'Name': 'Name'        # <--- new line
}

# Keep only columns that exist in the TSV
existing_cols = [c for c in rename_map.keys() if c in df.columns]
df_selected = df[existing_cols].copy()
df_selected.rename(columns=rename_map, inplace=True)

# Add placeholder columns that may be missing
if 'CHROM' not in df_selected.columns:
    df_selected['CHROM'] = '2'
if 'ID' not in df_selected.columns:
    df_selected['ID'] = ''
if 'QUAL' not in df_selected.columns:
    df_selected['QUAL'] = ''
if 'FILTER' not in df_selected.columns:
    df_selected['FILTER'] = 'PASS'
if 'CLNDISDB' not in df_selected.columns:
    df_selected['CLNDISDB'] = ''

# ============================================================
# 3. Clean REF and ALT: remove 'na' strings and empty values
# ============================================================
df_selected['REF'] = df_selected['REF'].astype(str).str.replace('na', '').str.strip()
df_selected['ALT'] = df_selected['ALT'].astype(str).str.replace('na', '').str.strip()
# Remove rows where both REF and ALT are empty
df_selected = df_selected[(df_selected['REF'] != '') | (df_selected['ALT'] != '')]
print(f"After cleaning REF/ALT: {len(df_selected)} rows")

# ============================================================
# 4. Create binary label from CLNSIG
# ============================================================
def get_label(clnsig):
    if pd.isna(clnsig):
        return np.nan
    s = str(clnsig).lower()
    if 'pathogenic' in s:
        return 1
    elif 'benign' in s:
        return 0
    else:
        return np.nan

df_selected['label'] = df_selected['CLNSIG'].apply(get_label)
initial_len = len(df_selected)
df_selected = df_selected.dropna(subset=['label'])
print(f"Dropped {initial_len - len(df_selected)} rows with ambiguous CLNSIG (VUS/conflicting)")

# ============================================================
# 5. Simplify CLNDN (take first disease if multiple)
# ============================================================
if 'CLNDN' in df_selected.columns:
    df_selected['CLNDN'] = df_selected['CLNDN'].astype(str).str.split('|').str[0]

# ============================================================
# 6. Ensure required columns are in the correct order (for feature_engineering.py)
# ============================================================
desired_order = ['CHROM', 'POS', 'ID', 'REF', 'ALT','Name', 'QUAL', 'FILTER',
                 'CLNSIG', 'CLNREVSTAT', 'MC', 'CLNDN', 'CLNVC',
                 'GENEINFO', 'CLNDISDB', 'ORIGIN', 'RS', 'label']

# Keep only columns that exist in df_selected
final_columns = [c for c in desired_order if c in df_selected.columns]
df_final = df_selected[final_columns]

# ============================================================
# 7. Save as bard1_variants.csv
# ============================================================
output_file = os.path.join(WORKSPACE_ROOT, "data", "intermediate", "bard1_variants.csv")
df_final.to_csv(output_file, index=False)
print(f"Saved {len(df_final)} variants to {output_file}")
print("Columns saved:", df_final.columns.tolist())
print("\nFirst 5 rows:")
print(df_final[['REF', 'ALT', 'CLNSIG', 'label']].head())