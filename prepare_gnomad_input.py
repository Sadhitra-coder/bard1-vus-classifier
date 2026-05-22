import pandas as pd

# Load your variant data
df = pd.read_csv("bard1_variants.csv")

# Ensure required columns exist
required = ['CHROM', 'POS', 'REF', 'ALT']
for col in required:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in CSV. Available: {df.columns.tolist()}")

# gnomAD expects CHROM without 'chr' prefix, and coordinates as integers
df['CHROM'] = df['CHROM'].astype(str).str.replace('chr', '')
df['POS'] = df['POS'].astype(int)



# Write to a text file (space-separated)
output_file = "variants_for_gnomad.txt"
df[['CHROM', 'POS', 'REF', 'ALT']].to_csv(output_file, sep=' ', index=False, header=False)

print(f"Saved {len(df)} variants to {output_file}")
print("First 5 lines:")
print(df[['CHROM', 'POS', 'REF', 'ALT']].head().to_string(index=False, header=False))