import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# Get the workspace root
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ============================================================
# 1. Load data
# ============================================================
data_file = os.path.join(WORKSPACE_ROOT, "data", "intermediate", "bard1_variants.csv")
df = pd.read_csv(data_file)
print(f"Loaded {len(df)} variants")
print("Columns:", df.columns.tolist())

# ============================================================
# 2. Basic feature engineering from REF/ALT
# ============================================================
df['ref_len'] = df['REF'].str.len()
df['alt_len'] = df['ALT'].str.len()
df['is_indel'] = (df['ref_len'] != df['alt_len']).astype(int)
df['indel_length'] = np.where(df['is_indel'] == 1,
                              abs(df['ref_len'] - df['alt_len']), 0)

def is_transition(ref, alt):
    if (ref in 'AG' and alt in 'AG') or (ref in 'CT' and alt in 'CT'):
        return 1
    return 0

df['is_transition'] = df.apply(lambda r: is_transition(r['REF'], r['ALT']), axis=1)
df['is_transversion'] = 1 - df['is_transition']

# ============================================================
# 3. Review status → numeric score
# ============================================================
def map_review_status(status_str):
    s = str(status_str).lower()
    if 'practice guideline' in s:
        return 4
    elif 'reviewed by expert panel' in s:
        return 3
    elif 'multiple submitters' in s:
        return 2
    elif 'single submitter' in s:
        return 1
    else:
        return 0

df['review_score'] = df['CLNREVSTAT'].apply(map_review_status)

# ============================================================
# 4. Consequence flags from MC (broad, will be overwritten later)
# ============================================================
mc = df['MC'].fillna('').astype(str).str.lower()
df['is_missense'] = mc.str.contains('missense', na=False).astype(int)
df['is_nonsense'] = mc.str.contains('nonsense|stop_gained', na=False).astype(int)
df['is_splice'] = mc.str.contains('splice', na=False).astype(int)
df['is_frameshift'] = mc.str.contains('frameshift', na=False).astype(int)
df['is_snv'] = mc.str.contains('single nucleotide variant', na=False).astype(int)

# ============================================================
# 5. Extract protein position and true consequence from Name column
# ============================================================
def parse_protein_change(name):
    """
    Extract protein position and consequence type from HGVS Name.
    Examples:
        NM_000465.4(BARD1):c.1670G>C (p.Cys557Ser)  -> (557, 'missense')
        NM_000465.4(BARD1):c.1690C>T (p.Gln564Ter)   -> (564, 'nonsense')
        NM_000465.4(BARD1):c.1935_1954dup (p.Glu652fs) -> (652, 'frameshift')
    """
    if pd.isna(name):
        return None, 'unknown'
    # Look for pattern like p.Cys557Ser or p.(Cys557Ser) or p.Gln564Ter or p.Glu652fs
    match = re.search(r'\(?p\.([A-Za-z]+)(\d+)([A-Za-z*]+)(?:fs)?\)?', name)
    if match:
        aa1, pos_str, aa2 = match.groups()
        pos = int(pos_str)
        if aa2 == '*' or 'Ter' in aa2:
            consequence = 'nonsense'
        elif 'fs' in name or 'frameshift' in name:
            consequence = 'frameshift'
        else:
            consequence = 'missense'
        return pos, consequence
    # Splice variants might not have protein change
    if 'splice' in name:
        return None, 'splice'
    return None, 'unknown'

df[['protein_pos', 'true_consequence']] = df['Name'].apply(
    lambda x: pd.Series(parse_protein_change(x))
)

# Overwrite consequence flags with the true ones (where available)
# Only for variants with protein position (i.e., coding variants)
has_protein = df['protein_pos'].notna()
df.loc[has_protein, 'is_missense'] = (df.loc[has_protein, 'true_consequence'] == 'missense').astype(int)
df.loc[has_protein, 'is_nonsense'] = (df.loc[has_protein, 'true_consequence'] == 'nonsense').astype(int)
df.loc[has_protein, 'is_frameshift'] = (df.loc[has_protein, 'true_consequence'] == 'frameshift').astype(int)
df.loc[has_protein, 'is_splice'] = (df.loc[has_protein, 'true_consequence'] == 'splice').astype(int)

# ============================================================
# 6. Protein domain mapping (BARD1 specific)
# ============================================================
def get_domain(pos):
    if 46 <= pos <= 90:
        return 'RING'
    elif 420 <= pos <= 555:
        return 'ANK'
    elif 568 <= pos <= 777:
        return 'BRCT'
    else:
        return 'other'

df['domain'] = df['protein_pos'].apply(lambda x: get_domain(x) if pd.notna(x) else 'other')
df['in_RING'] = (df['domain'] == 'RING').astype(int)
df['in_BRCT'] = (df['domain'] == 'BRCT').astype(int)
df['in_ANK'] = (df['domain'] == 'ANK').astype(int)
df['in_domain'] = (df['domain'] != 'other').astype(int)

# ============================================================
# 7. Disease flag (breast cancer)
# ============================================================
df['has_breast_cancer'] = df['CLNDN'].astype(str).str.contains('breast|cancer', case=False, na=False).astype(int)

# ============================================================
# 8. Germline flag
# ============================================================
df['is_germline'] = df['ORIGIN'].astype(str).str.contains('germline', case=False, na=False).astype(int)

# ============================================================
# 9. Drop non‑predictive columns (identifiers, raw text, coordinates)
# ============================================================
drop_cols = ['ID', 'QUAL', 'FILTER', 'GENEINFO', 'CLNDISDB', 'RS', 'CHROM', 'POS',
             'REF', 'ALT', 'CLNSIG', 'CLNREVSTAT', 'MC', 'CLNDN', 'ORIGIN',
             'Name', 'protein_pos', 'true_consequence', 'domain']
df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

# ============================================================
# 10. Drop any rows where label is missing
# ============================================================
df_clean = df_clean.dropna(subset=['label'])
print(f"After dropping missing labels: {len(df_clean)} rows")

# ============================================================
# 11. EDA: Missing values, duplicates, class balance
# ============================================================
print("\n=== Missing values per column ===")
print(df_clean.isnull().sum())

print(f"\n=== Duplicate rows: {df_clean.duplicated().sum()} ===")

print("\n=== Class distribution ===")
print(df_clean['label'].value_counts(normalize=True))

# ============================================================
# 12. Histograms for numeric features
# ============================================================
numeric_cols = ['ref_len', 'alt_len', 'indel_length', 'review_score']
numeric_cols = [c for c in numeric_cols if c in df_clean.columns]
if numeric_cols:
    df_clean[numeric_cols].hist(figsize=(12, 8), bins=30)
    plt.suptitle("Feature Distributions")
    plt.tight_layout()
    histogram_path = os.path.join(WORKSPACE_ROOT, "results", "feature_histograms.png")
    plt.savefig(histogram_path, dpi=150)
    print(f"Saved histograms to {histogram_path}")
    plt.show()
else:
    print("No numeric columns for histograms.")

# ============================================================
# 13. Bar plots for binary features
# ============================================================
binary_cols = ['is_indel', 'is_transition', 'is_missense', 'is_nonsense',
               'is_splice', 'is_frameshift', 'is_snv', 'has_breast_cancer',
               'is_germline', 'in_RING', 'in_BRCT', 'in_ANK', 'in_domain']
binary_cols = [c for c in binary_cols if c in df_clean.columns]
if binary_cols:
    n_cols = len(binary_cols)
    n_rows = (n_cols + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols > 1 else [axes]
    for i, col in enumerate(binary_cols):
        df_clean[col].value_counts().plot(kind='bar', ax=axes[i], title=col, color='skyblue')
        axes[i].set_xlabel('0 / 1')
        axes[i].set_ylabel('Count')
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    barplot_path = os.path.join(WORKSPACE_ROOT, "results", "binary_feature_barplots.png")
    plt.savefig(barplot_path, dpi=150)
    print(f"Saved bar plots to {barplot_path}")
    plt.show()
else:
    print("No binary columns for bar plots.")

# ============================================================
# 14. Correlation heatmap (numeric columns only)
# ============================================================
numeric_df = df_clean.select_dtypes(include='number')
if not numeric_df.empty and numeric_df.shape[1] > 1:
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    heatmap_path = os.path.join(WORKSPACE_ROOT, "results", "correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    print(f"Saved correlation heatmap to {heatmap_path}")
    plt.show()
else:
    print("Not enough numeric columns for correlation heatmap.")

# ============================================================
# 15. Save cleaned feature set for model training
# ============================================================
output_file = os.path.join(WORKSPACE_ROOT, "data", "processed", "bard1_features_cleaned.csv")
df_clean.to_csv(output_file, index=False)
print(f"\nSaved cleaned dataset to {output_file}")
print(f"Shape: {df_clean.shape}")
print("Features:", df_clean.columns.tolist())