"""Convert protein embeddings to gene embeddings by averaging the protein embeddings for each gene.

This must be run inside the Python 3.10 saturn conda environment: source ~/miniconda3/bin/activate && conda activate saturn
"""
import os
import json
from pathlib import Path
from typing_extensions import Literal
import glob

import torch
from tap import Tap
from tqdm import tqdm


fasta_root_folder = Path("/mnt/data/projects/cell_atlas_approximations/reference_atlases/data/saturn/peptide_sequences/")
embedding_root_fdn = fasta_root_folder.parent / "esm_embeddings"
output_fdn = embedding_root_fdn.parent / "esm_embeddings_summaries/"
os.makedirs(output_fdn, exist_ok=True)


# Last layer of pretrained transformer
LAST_LAYER = 33 # ESM1b
MSA_LAST_LAYER = 12 # MSA
LAST_LAYER_2 = 48 # ESM2


def infer_model(embedding_dir):
    model = str(embedding_dir).split('_')[-1].upper()
    model = model[:-1] + model[-1].lower()
    return model


def summarize_gene_embeddings(subfdn) -> None:
    """Convert protein embeddings to gene embeddings by averaging the protein embeddings for each gene."""

    embedding_dir = embedding_root_fdn / subfdn
    embedding_model = infer_model(
        embedding_dir=embedding_dir
    )
    embedding_model_lower = embedding_model.lower()

    species = Path(subfdn).stem
    print(species)
    output_fn = output_fdn / f"{species}_gene_all_{embedding_model_lower}.pt"
    print(output_fn)

    # Get last layer
    if embedding_model == 'ESM1b':
        last_layer = LAST_LAYER
    elif embedding_model == 'MSA1b':
        last_layer = MSA_LAST_LAYER
    elif embedding_model == 'ESM2':
        last_layer = LAST_LAYER_2
    else:
        raise ValueError(f'Embedding model "{embedding_model}" is not supported.')

    # Get protein embedding paths
    protein_embedding_paths = glob.glob(str(embedding_dir) + "/*.pt")

    # Create mapping from gene name to embedding, considering the proteins are already representatives
    # NOTE: This differs from the original SATURN implementation, which averages the embeddings of all isoforms
    # within each gene, with equal weights across isoforms.
    gene_symbol_to_embedding = {}
    for protein_embedding_path in tqdm(protein_embedding_paths):
        gene = Path(protein_embedding_path).stem
        embedding = torch.load(protein_embedding_path)['mean_representations'][last_layer]
        gene_symbol_to_embedding[gene] = embedding

    genes = list(gene_symbol_to_embedding.keys())
    print(genes[:10])

    # Save gene symbol to embedding map
    torch.save(gene_symbol_to_embedding, output_fn)


if __name__ == '__main__':

    for subfdn in os.listdir(embedding_root_fdn):
        summarize_gene_embeddings(subfdn)
