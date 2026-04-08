import modal
from pydantic import BaseModel


class VariantRequest(BaseModel):
    variant_position: int
    alternative: str
    genome: str
    chromosome: str

evo2_image = modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12").apt_install(
    "build-essential",
    "cmake",
    "ninja-build",
    "libcudnn8-dev",
    "libcudnn8",
    "git",
    "gcc",
    "g++"
).env({
    "CC": "/usr/bin/gcc",
    "CXX": "/usr/bin/g++",
    "NIM_VARIANT": "7b"
}).run_commands(
    "pip install --upgrade pip setuptools wheel",
    "pip install torch==2.8.0",
    "pip install flash-attn --no-build-isolation",
    "pip uninstall -y transformer-engine transformer_engine",
    "pip install 'transformer_engine[pytorch]>=2.0.0' --no-build-isolation",
    "git clone https://github.com/arcinstitute/evo2 && cd evo2 && pip install -e .",
).pip_install_from_requirements("requirements.txt")

app = modal.App("variant-analysis-evo2", image=evo2_image)
volume = modal.Volume.from_name("evo2-volume", create_if_missing=True)
mount_path = "/root/.cache/huggingface"

@app.function(gpu="h100", volumes={mount_path: volume}, timeout=1000)
def run_brca_analysis():
    from Bio import SeqIO
    import base64
    from io import BytesIO
    import gzip
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os
    import seaborn as sns
    from sklearn.metrics import roc_auc_score
    from evo2 import Evo2
    WINDOW_SIZE = 8192
    print("Running BRCA analysis")
    print("Loading evo2 model")

    model = Evo2('evo2_7b')
    print("Evo2 model loaded")

    brca1_df = pd.read_excel(
            os.path.join('/evo2/notebooks/brca1/41586_2018_461_MOESM3_ESM.xlsx'),
            header=2,
    )
    brca1_df = brca1_df[[
        'chromosome', 'position (hg19)', 'reference', 'alt', 'function.score.mean', 'func.class',
    ]]
    # Rename columns
    brca1_df.rename(columns={
        'chromosome': 'chrom',
        'position (hg19)': 'pos',
        'reference': 'ref',
        'alt': 'alt',
        'function.score.mean': 'score',
        'func.class': 'class',
    }, inplace=True)

    # Convert to two-class system
    brca1_df['class'] = brca1_df['class'].replace(['FUNC', 'INT'], 'FUNC/INT')


    # Read the reference genome sequence of chromosome 17
    with gzip.open(os.path.join('/evo2/notebooks/brca1/GRCh37.p13_chr17.fna.gz'), "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq_chr17 = str(record.seq)
            break
    ref_seqs = []
    ref_seq_to_index = {}

    # Parse sequences and store indexes
    ref_seq_indexes = []
    var_seqs = []

    brca1_subset=brca1_df.iloc[:200].copy()

    for _, row in brca1_subset.iterrows():
        pos = row['pos']
        ref = row['ref']
        alt = row['alt']
        p = pos - 1  # Convert to 0-indexed position
        full_seq = seq_chr17

        ref_seq_start = max(0, p - WINDOW_SIZE//2)
        ref_seq_end = min(len(full_seq), p + WINDOW_SIZE//2)
        ref_seq = seq_chr17[ref_seq_start:ref_seq_end]
        snv_pos_in_ref = min(WINDOW_SIZE//2, p)
        var_seq = ref_seq[:snv_pos_in_ref] + alt + ref_seq[snv_pos_in_ref+1:]


        # Get or create index for reference sequence
        if ref_seq not in ref_seq_to_index:
            ref_seq_to_index[ref_seq] = len(ref_seqs)
            ref_seqs.append(ref_seq)

        ref_seq_indexes.append(ref_seq_to_index[ref_seq])
        var_seqs.append(var_seq)

    ref_seq_indexes = np.array(ref_seq_indexes)

    print(
        f'Scoring likelihoods of {len(ref_seqs)} reference sequences with Evo 2...')
    ref_scores = model.score_sequences(ref_seqs)

    print(
        f'Scoring likelihoods of {len(var_seqs)} variant sequences with Evo 2...')
    var_scores = model.score_sequences(var_seqs)
    delta_scores = np.array(var_scores) - np.array(ref_scores)[ref_seq_indexes]


    # Add delta scores to dataframe
    brca1_subset[f'evo2_delta_score'] = delta_scores
    
    y_true = (brca1_subset['class'] == 'LOF')
    auroc = roc_auc_score(y_true, -brca1_subset['evo2_delta_score'])

    plt.figure(figsize=(4, 2))


    # Plot stripplot of distributions
    p = sns.stripplot(
        data=brca1_subset,
        x='evo2_delta_score',
        y='class',
        hue='class',
        order=['FUNC/INT', 'LOF'],
        palette=['#777777', 'C3'],
        size=2,
        jitter=0.3,
    )

    # Mark medians from each distribution
    sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'visible': False},
                medianprops={'color': 'k', 'ls': '-', 'lw': 2},
                whiskerprops={'visible': False},
                zorder=10,
                x="evo2_delta_score",
                y="class",
                data=brca1_subset,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=p)
    plt.xlabel('Delta likelihood score, Evo 2')
    plt.ylabel('BRCA1 SNV class')
    plt.tight_layout()

    #send to local machine in base64 encoded image
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    return {"variants": brca1_subset.to_dict(orient='records'), "plot": plot_base64, "auroc": auroc}

@app.function()
def deploy():
    import base64
    from io import BytesIO
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    results= run_brca_analysis.remote()

    if "plot" in results:
        plot_base64 = results["plot"]
        plot_bytes = base64.b64decode(plot_base64)
        with open("brca1_delta_scores.png", "wb") as f:
            f.write(plot_bytes)
        img = mpimg.imread(BytesIO(plot_bytes))
        plt.figure(figsize=(10, 5))
        plt.imshow(img)
        plt.axis('off')
        plt.show()


    if "auroc" in results:
        auroc = results["auroc"]
        print(f'Zero-shot prediction AUROC: {auroc:.2}')
    # print("Deployed")


def get_genome_sequence(position, genome: str, chromosome: str, window_size=8192):
    import requests

    half_window = window_size // 2
    start = max(0, position - 1 - half_window)
    end = position - 1 + half_window + 1

    print(
        f"Fetching {window_size}bp window around position {position} from UCSC API..")
    print(f"Coordinates: {chromosome}:{start}-{end} ({genome})")

    api_url = f"https://api.genome.ucsc.edu/getData/sequence?genome={genome};chrom={chromosome};start={start};end={end}"
    response = requests.get(api_url)

    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch genome sequence from UCSC API: {response.status_code}")

    genome_data = response.json()

    if "dna" not in genome_data:
        error = genome_data.get("error", "Unknown error")
        raise Exception(f"UCSC API errpr: {error}")

    sequence = genome_data.get("dna", "").upper()
    expected_length = end - start
    if len(sequence) != expected_length:
        print(
            f"Warning: received sequence length ({len(sequence)}) differs from expected ({expected_length})")

    print(
        f"Loaded reference genome sequence window (length: {len(sequence)} bases)")

    return sequence, start


def analyze_variant(relative_pos_in_window, reference, alternative, window_seq, model):
    var_seq = window_seq[:relative_pos_in_window] + \
        alternative + window_seq[relative_pos_in_window+1:]

    print(f"=== Variant Analysis Debug ===")
    print(f"Reference nucleotide at position: {reference}")
    print(f"Alternative nucleotide: {alternative}")
    print(f"Relative position in window: {relative_pos_in_window}")
    print(f"Window sequence length: {len(window_seq)}")
    print(f"Variant sequence length: {len(var_seq)}")
    print(f"Context around variant (ref): ...{window_seq[max(0, relative_pos_in_window-10):relative_pos_in_window+11]}...")
    print(f"Context around variant (var): ...{var_seq[max(0, relative_pos_in_window-10):relative_pos_in_window+11]}...")
    print(f"Sequences are identical: {window_seq == var_seq}")

    ref_score = model.score_sequences([window_seq])[0]
    var_score = model.score_sequences([var_seq])[0]

    delta_score = var_score - ref_score
    
    print(f"Reference score: {ref_score}")
    print(f"Variant score: {var_score}")
    print(f"Delta score: {delta_score}")
    print(f"=== End Debug ===")

    threshold = -0.0009178519
    lof_std = 0.0015140239
    func_std = 0.0009016589

    if delta_score < threshold:
        prediction = "Likely pathogenic"
        confidence = min(1.0, abs(delta_score - threshold) / lof_std)
    else:
        prediction = "Likely benign"
        confidence = min(1.0, abs(delta_score - threshold) / func_std)

    return {
        "reference": reference,
        "alternative": alternative,
        "delta_score": float(delta_score),
        "prediction": prediction,
        "classification_confidence": float(confidence)
    }


@app.cls(gpu="h100", volumes={mount_path: volume}, max_containers=3, retries=2, scaledown_window=120)
class Evo2Model:
    @modal.enter()
    def load_evo2(self):
        from evo2 import Evo2
        self.model = Evo2('evo2_7b')
        print("Evo2 model loaded")


    @modal.fastapi_endpoint(method="POST")
    def analyze_single_variant(self, request: VariantRequest):
        variant_position = request.variant_position
        alternative = request.alternative
        genome = request.genome
        chromosome = request.chromosome

        print("=" * 50)
        print("RECEIVED REQUEST:")
        print(f"  Genome: {genome} (type: {type(genome)})")
        print(f"  Chromosome: {chromosome} (type: {type(chromosome)})")
        print(f"  Variant position: {variant_position} (type: {type(variant_position)})")
        print(f"  Variant alternative: '{alternative}' (type: {type(alternative)})")
        print("=" * 50)

        WINDOW_SIZE = 8192

        window_seq, seq_start = get_genome_sequence(
            position=variant_position,
            genome=genome,
            chromosome=chromosome,
            window_size=WINDOW_SIZE
            )

        print(f"Fetched genome seauence window, first 100: {window_seq[:100]}")

        relative_pos = variant_position - 1 - seq_start
        print(f"Relative position within window: {relative_pos}")

        if relative_pos < 0 or relative_pos >= len(window_seq):
            raise ValueError(
                f"Variant position {variant_position} is outside the fetched window (start={seq_start+1}, end={seq_start+len(window_seq)})")

        reference = window_seq[relative_pos]
        print("Reference is: " + reference)

        # Analyze the variant
        result = analyze_variant(
            relative_pos_in_window=relative_pos,
            reference=reference,
            alternative=alternative,
            window_seq=window_seq,
            model=self.model
        )

        result["position"] = variant_position

        return result


@app.local_entrypoint()
def main():
    # Example of how you'd call the deployed Modal Function from your client
    import requests
    import json    # brca1_example.remote()

    evo2Model = Evo2Model()

    url = evo2Model.analyze_single_variant.web_url

    payload = {
        "variant_position": 43119628,
        "alternative": "G",
        "genome": "hg38",
        "chromosome": "chr17"
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    result = response.json()
    print(result)

        



# @app.local_entrypoint()
# def main():
#     # deploy.remote()
#     evo2_model = Evo2Model()
#     evo2_model.analyse_variant(variant_position=4319628, alternative='G', genome_assembly='hg38', chromosome='chr17')
