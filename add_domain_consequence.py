import pandas as pd
import re

# Load original variant file (has Name column with HGVS)
df = pd.read_csv("bard1_variants.tsv", sep='\t')
print("Original columns:", df.columns.tolist())

# The Name column might be named 'Name' or 'variant_name'? Check.
if 'Name' not in df.columns:
    # maybe it's 'variant_name' from earlier? Let's look for it
    if 'variant_name' in df.columns:
        df.rename(columns={'variant_name': 'Name'}, inplace=True)
    else:
        print("No Name column found. Please provide column with HGVS protein change.")
        exit()

# Extract protein position and consequence type from Name
def parse_hgvs(name):
    """Extract protein position and consequence type from HGVS name."""
    # Example: "NM_000465.4(BARD1):c.1670G>C (p.Cys557Ser)"
    # Look for p. pattern
    p_match = re.search(r'p\.\(?([A-Za-z]+)(\d+)([A-Za-z]+)(?:fs)?\)?', name)
    if p_match:
        aa1, pos, aa2 = p_match.groups()
        position = int(pos)
        # Determine consequence
        if aa2 == '*':
            consequence = 'nonsense'
        elif 'fs' in name or 'frameshift' in name:
            consequence = 'frameshift'
        else:
            consequence = 'missense'
        return position, consequence
    # Check for nonsense without p. (e.g., "p.Gln564Ter")
    ter_match = re.search(r'p\.\(?([A-Za-z]+)(\d+)Ter\)?', name)
    if ter_match:
        _, pos, _ = ter_match.groups()
        return int(pos), 'nonsense'
    # Check for splice or other
    if 'splice' in name:
        return None, 'splice'
    if 'frameshift' in name:
        return None, 'frameshift'
    return None, 'unknown'

df[['protein_pos', 'consequence']] = df['Name'].apply(lambda x: pd.Series(parse_hgvs(x)))

# Drop rows without protein position (e.g., intronic, UTR, etc.)
df = df.dropna(subset=['protein_pos'])
print(f"Rows with protein position: {len(df)}")

# Domain boundaries for BARD1 (from UniProt)
def get_domain(pos):
    if 46 <= pos <= 90:
        return 'RING'
    elif 420 <= pos <= 555:
        return 'ANK'
    elif 568 <= pos <= 777:
        return 'BRCT'
    else:
        return 'other'

df['domain'] = df['protein_pos'].apply(get_domain)

# Create binary domain flags
df['in_RING'] = (df['domain'] == 'RING').astype(int)
df['in_BRCT'] = (df['domain'] == 'BRCT').astype(int)
df['in_ANK'] = (df['domain'] == 'ANK').astype(int)
df['in_domain'] = (df['domain'] != 'other').astype(int)

# Create consequence flags if not already present (will replace previous ones)
df['is_missense'] = (df['consequence'] == 'missense').astype(int)
df['is_nonsense'] = (df['consequence'] == 'nonsense').astype(int)
df['is_frameshift'] = (df['consequence'] == 'frameshift').astype(int)
df['is_splice'] = (df['consequence'] == 'splice').astype(int)

# Keep only relevant columns for merging with existing features
domain_cols = ['CHROM', 'POS', 'REF', 'ALT', 'label', 'protein_pos', 'domain',
               'in_RING', 'in_BRCT', 'in_ANK', 'in_domain',
               'is_missense', 'is_nonsense', 'is_frameshift', 'is_splice']
df_domain = df[domain_cols].copy()
print(df_domain.head())

# Save this file for merging
df_domain.to_csv("bard1_domain_features.csv", index=False)
print("Saved domain features to bard1_domain_features.csv")