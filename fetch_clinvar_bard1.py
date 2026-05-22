import gzip
import csv
import pandas as pd
import numpy as np


input_file = "variant_summary.txt.gz"
output_file = "bard1_small.tsv"

with gzip.open(input_file, 'rt') as infile, open(output_file, 'w') as outfile:
    reader = csv.DictReader(infile, delimiter='\t')
    # Get fieldnames from the reader
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()
    
    count = 0
    for row in reader:
        # Check conditions using column names (much safer)
        if (row.get('GeneSymbol') == 'BARD1' and
            'copy number' not in row.get('VariantType', '') and
            ('Pathogenic' in row.get('ClinicalSignificance', '') or
             'Benign' in row.get('ClinicalSignificance', ''))):
            writer.writerow(row)
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} matching rows...")
    
    print(f"Done. Saved {count} variants to {output_file}")
    