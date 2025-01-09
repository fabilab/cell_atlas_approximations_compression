"""Test inference with SATURN-like model on the termites/fly data.

This must be run inside the Python 3.10 saturn conda environment: source ~/miniconda3/bin/activate && conda activate saturn

"""
import os
import pathlib
import pandas as pd
import subprocess as sp
import argparse



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train SATURN on the atlasapprox data.")
    parser.add_argument('--dry', action="store_true", help="Dry run")
    parser.add_argument('--n-macrogenes', default=6, help="Number of macrogenes (only used to find the model)")
    parser.add_argument('--n-hvg', default=12, help="Number of highly variable genes")
    parser.add_argument('--n-epochs', default=1, help="Number of epochs in metric learning")
    parser.add_argument('--n-pretrain-epochs', default=1, help="Number of epochs in pretraining")
    parser.add_argument('--species', type=str, help="Infer embedding for data of this species", default="a_queenslandica")
    parser.add_argument('--train', action='store_true', help="Whether to train the transfer model")
    args = parser.parse_args()

    fasta_root_folder = pathlib.Path("/mnt/data/projects/cell_atlas_approximations/reference_atlases/data/saturn/peptide_sequences/")
    embedding_root_fdn = fasta_root_folder.parent / "esm_embeddings"
    embeddings_summary_fdn = embedding_root_fdn.parent / "esm_embeddings_summaries/"
    h5ad_fdn = embeddings_summary_fdn.parent / "h5ads"
    training_output_fdn = embeddings_summary_fdn.parent / f"output_nmacro{args.n_macrogenes}_nhvg{args.n_hvg}_epochs_p{args.n_pretrain_epochs}_m{args.n_epochs}"
    centroids_fn = training_output_fdn / "centroids.pkl"  # This is temp output to speed up later iterations (kmeans is costly, apparently)
    pretrain_model_fn = training_output_fdn /"pretrain_model.model"
    metric_model_fn = training_output_fdn /"metric_model.model"
    output_fdn = training_output_fdn / "inference_output"
    training_csv_fn = training_output_fdn / "in_csv.csv"
    trained_adata_path = training_output_fdn / "saturn_results" / "final_adata.h5ad"

    # Build the CSV used by SATURN to connect the species
    for h5ad_fn in h5ad_fdn.iterdir():
        species = h5ad_fn.stem
        if species != args.species:
            continue
        print(species)
        embedding_summary_fn = embeddings_summary_fdn / f"{species}_gene_all_esm1b.pt"
        if not embedding_summary_fn.exists():
            print(" Embedding summary file not found, skipping")
            continue
        break
    else:
        raise IOError("adata or embedding summary files not found")

    # Sanity check: verify all features used in the h5ad var_names have a corresponding embedding
    row = {"species": species, "path": h5ad_fn, "embedding_path": embedding_summary_fn}
    print("Checking", row["species"])
    adata = __import__("anndata").read_h5ad(row["path"])
    embedding = __import__("torch").load(row["embedding_path"])
    features_h5ad = adata.var_names
    features_emb = pd.Index(embedding.keys())
    features_xor = set(features_h5ad) ^ set(features_emb)
    if len(features_xor) > 0:
        if len(features_xor) < 10:
            nfea = len(features_xor)
            print(f"Features in h5ad but not in embedding ({nfea}), correcting h5ad file:", features_xor)
            adata = adata[:, features_emb]
            adata.write_h5ad(row["path"])
        else:
            assert features_h5ad.isin(features_emb).all()
    del adata, embedding
    __import__("gc").collect()

    # Run SATURN inference
    script_fn = pathlib.Path("/home/fabio/projects/termites") / "software" / "SATURN" / "inference.py"
    #scoring_maps_fn = output_fdn / "scoring_cell_type_maps.csv"  # This is temp output to speed up later iterations (kmeans is costly, apparently)

    call = [
        "python",
        str(script_fn),
        f"--in_adata_path={h5ad_fn}",
        f"--in_embeddings_path={embedding_summary_fn}",
        "--in_label_col=cellType",
        "--ref_label_col=cellType",
        f"--centroids_init_path={centroids_fn}",
        f"--pretrain_model_path={pretrain_model_fn}",
        f"--metric_model_path={metric_model_fn}",
        f"--work_dir={output_fdn}",  # This is general output
        f"--hv_genes={args.n_hvg}",
        "--seed=42",
        f"--species={args.species}-inf",
        f"--training_csv_path={training_csv_fn}",
    ]
    if args.train:
        call += [
            '--train',
            f'--trained_adata_path={trained_adata_path}',
        ]
    print(" ".join(call))
    if not args.dry:
        sp.run(" ".join(call), shell=True, check=True)
