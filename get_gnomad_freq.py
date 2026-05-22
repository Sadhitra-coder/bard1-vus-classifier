import requests
import csv
import time
import math

def fetch_variant_batch(variant_ids):
    """
    Fetch gnomAD v4 details for multiple variants in one request.
    variant_ids: list of strings like "2-214752454-C-G"
    Returns dict mapping variant_id -> data.
    """
    # Build a GraphQL query with multiple variant queries
    queries = []
    for vid in variant_ids:
        queries.append(f"""
        v{hash(vid) % 1000000}: variant(variantId: "{vid}", dataset: gnomad_r4) {{
            variant_id
            chrom
            pos
            ref
            alt
            joint {{
                ac
                an
                homozygote_count
            }}
        }}
        """)
    batch_query = "{" + " ".join(queries) + "}"

    response = requests.post(
        'https://gnomad.broadinstitute.org/api',
        json={'query': batch_query}
    )
    if response.status_code == 200:
        return response.json().get('data', {})
    else:
        raise Exception(f"Batch query failed: {response.status_code}")

if __name__ == "__main__":
    # Read variants
    with open("variants_for_gnomad.txt", "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    batch_size = 20  # number of variants per request (adjust down if errors)
    results = []

    for i in range(0, len(lines), batch_size):
        batch = lines[i:i+batch_size]
        variant_ids = []
        for line in batch:
            parts = line.split()
            if len(parts) != 4:
                continue
            chrom, pos, ref, alt = parts
            variant_ids.append(f"{chrom}-{pos}-{ref}-{alt}")

        print(f"Batch {i//batch_size + 1}/{(len(lines)+batch_size-1)//batch_size} ({len(variant_ids)} variants)")

        success = False
        for attempt in range(5):
            try:
                data = fetch_variant_batch(variant_ids)
                # Map back to variant_id
                for vid in variant_ids:
                    variant_data = data.get(f"v{hash(vid)%1000000}", {})
                    if variant_data:
                        joint = variant_data.get('joint', {})
                        af = joint.get('ac', 0) / joint.get('an', 1) if joint.get('an') else None
                        results.append({
                            'variant_id': vid,
                            'chrom': variant_data.get('chrom'),
                            'pos': variant_data.get('pos'),
                            'ref': variant_data.get('ref'),
                            'alt': variant_data.get('alt'),
                            'joint_af': af,
                            'joint_ac': joint.get('ac'),
                            'joint_an': joint.get('an'),
                            'joint_hom': joint.get('homozygote_count')
                        })
                    else:
                        results.append({'variant_id': vid, 'joint_af': None})
                success = True
                break
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
                if "429" in str(e):
                    wait = 10 * (attempt+1)
                    print(f"Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    time.sleep(2)
        if not success:
            print("Batch failed after retries, skipping")
            for vid in variant_ids:
                results.append({'variant_id': vid, 'joint_af': None})

        time.sleep(1)  # be polite

    # Write CSV
    with open("gnomad_frequencies.csv", "w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=['variant_id', 'chrom', 'pos', 'ref', 'alt', 'joint_af', 'joint_ac', 'joint_an', 'joint_hom'])
        writer.writeheader()
        writer.writerows(results)

    print(f"Done. Results saved to gnomad_frequencies.csv")