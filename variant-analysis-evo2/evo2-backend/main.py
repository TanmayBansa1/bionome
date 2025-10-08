import sys

import modal

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

# app = modal.App.lookup('variant-analysis-evo2', create_if_missing=True)

# sb = modal.Sandbox.create(app=app, gpu="h100")
# p = sb.exec("echo 'Hello, World!'")
# print(p.stdout.read())
# sb.terminate()
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


@app.local_entrypoint()
def main():
    # deploy.remote()
    deploy.local()
