import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ============================================================
# 1. Load data
# ============================================================
df = pd.read_csv("bard1_variants.csv")
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
# 3. Review status → numeric score (fuzzy matching)
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
# 4. Consequence flags from MC (which contains Type e.g., "single nucleotide variant")
# ============================================================
mc = df['MC'].fillna('').astype(str).str.lower()

df['is_missense'] = mc.str.contains('missense', na=False).astype(int)
df['is_nonsense'] = mc.str.contains('nonsense|stop_gained', na=False).astype(int)
df['is_splice'] = mc.str.contains('splice', na=False).astype(int)
df['is_frameshift'] = mc.str.contains('frameshift', na=False).astype(int)

# For SNV flag, check if MC contains 'single nucleotide variant'
df['is_snv'] = mc.str.contains('single nucleotide variant', na=False).astype(int)

# ============================================================
# 5. Disease flag (breast cancer)
# ============================================================
df['has_breast_cancer'] = df['CLNDN'].astype(str).str.contains('breast|cancer', case=False, na=False).astype(int)

# ============================================================
# 6. Germline flag
# ============================================================
df['is_germline'] = df['ORIGIN'].astype(str).str.contains('germline', case=False, na=False).astype(int)

# ============================================================
# 7. Drop non‑predictive columns (identifiers, raw text)
# ============================================================
drop_cols = ['ID', 'QUAL', 'FILTER', 'GENEINFO', 'CLNDISDB', 'RS', 'CHROM', 'POS',
             'REF', 'ALT', 'CLNSIG', 'CLNREVSTAT', 'MC', 'CLNDN', 'ORIGIN']
df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

# ============================================================
# 8. Drop any rows where label is missing
# ============================================================
df_clean = df_clean.dropna(subset=['label'])
print(f"After dropping missing labels: {len(df_clean)} rows")

# ============================================================
# 9. EDA: Missing values, duplicates, class balance
# ============================================================
print("\n=== Missing values per column ===")
print(df_clean.isnull().sum())

print(f"\n=== Duplicate rows: {df_clean.duplicated().sum()} ===")

print("\n=== Class distribution ===")
print(df_clean['label'].value_counts(normalize=True))

# ============================================================
# 10. Histograms for numeric features
# ============================================================
numeric_cols = ['ref_len', 'alt_len', 'indel_length', 'review_score']
numeric_cols = [c for c in numeric_cols if c in df_clean.columns]
if numeric_cols:
    df_clean[numeric_cols].hist(figsize=(12, 8), bins=30)
    plt.suptitle("Feature Distributions")
    plt.tight_layout()
    plt.savefig("feature_histograms.png", dpi=150)
    plt.show()
else:
    print("No numeric columns for histograms.")

# ============================================================
# 11. Bar plots for binary features
# ============================================================
binary_cols = ['is_indel', 'is_transition', 'is_missense', 'is_nonsense',
               'is_splice', 'is_frameshift', 'is_snv', 'has_breast_cancer', 'is_germline']
binary_cols = [c for c in binary_cols if c in df_clean.columns]
if binary_cols:
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for i, col in enumerate(binary_cols):
        df_clean[col].value_counts().plot(kind='bar', ax=axes[i], title=col, color='skyblue')
        axes[i].set_xlabel('0 / 1')
        axes[i].set_ylabel('Count')
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig("binary_feature_barplots.png", dpi=150)
    plt.show()
else:
    print("No binary columns for bar plots.")

# ============================================================
# 12. Correlation heatmap (numeric columns only)
# ============================================================
numeric_df = df_clean.select_dtypes(include='number')
if not numeric_df.empty and numeric_df.shape[1] > 1:
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    plt.show()
else:
    print("Not enough numeric columns for correlation heatmap.")

# ============================================================
# 13. Save cleaned feature set for model training
# ============================================================
output_file = "bard1_features_cleaned.csv"
df_clean.to_csv(output_file, index=False)
print(f"\nSaved cleaned dataset to {output_file}")
print(f"Shape: {df_clean.shape}")
print("Features:", df_clean.columns.tolist())