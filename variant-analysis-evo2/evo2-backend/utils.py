import modal
import os
import requests
from main import app
WINDOW_SIZE = 8192
def parse_sequences(pos, ref_seq, alt):
    """
        Parse reference and variant sequences from the reference genome sequence.
        """
    p = pos - 1  # Convert to 0-indexed position
    full_seq = ref_seq

    ref_seq_start = max(0, p - WINDOW_SIZE//2)
    ref_seq_end = min(len(full_seq), p + WINDOW_SIZE//2)
    ref_seq = ref_seq[ref_seq_start:ref_seq_end]
    snv_pos_in_ref = min(WINDOW_SIZE//2, p)
    var_seq = ref_seq[:snv_pos_in_ref] + alt + ref_seq[snv_pos_in_ref+1:]

    return ref_seq, var_seq

@app.function(secrets=[modal.Secret.from_name('custom_secret')])
def get_ref_sequence(position: int, genome_sequence: str, chromosome: str):
    # make an api call to the ucsc genome api to get the reference sequence

    ref_start = max(0, position - WINDOW_SIZE//2 - 1)
    ref_end = min(len(genome_sequence), position + WINDOW_SIZE//2)

    ucsc_base_url = os.environ('UCSC_GENOME_API_BASE_URL')
    api_url=f"{ucsc_base_url}/getData/sequence?genome={genome_sequence};chrom={chromosome};start={ref_start};end={ref_end}"

    response = requests.get(api_url)
    if response.status_code != 200:
        raise Exception(f"Failed to get reference sequence from UCSC genome api: {response.status_code}")
    data = response.json()
    if data.get('error') or 'dna' not in data:
        raise Exception(f"Failed to get reference sequence from UCSC genome api: {data.get('error')}")
    ref_seq = data.get('dna').upper()

    # sanity checks
    assert len(ref_seq) == ref_end - ref_start
    print(f'Reference sequence loaded from UCSC genome api')

    return ref_seq
    
    
    