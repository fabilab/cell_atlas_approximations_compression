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
    parser.add_argument('--n-hvg', default=13, help="Number of highly variable genes")
    parser.add_argument('--n-epochs', default=1, help="Number of epochs in metric learning")
    parser.add_argument('--n-pretrain-epochs', default=1, help="Number of epochs in pretraining")
    parser.add_argument('--species', type=str, help="Infer embedding for data of this species", default="a_queenslandica")
    parser.add_argument('--train', action='store_true', help="Whether to train the transfer model")
    parser.add_argument('--leaveout', type=str, default=None, help="Use leaveout-traned model without this species for inference.")
    parser.add_argument('--secondary-analysis', action='store_true', help="Perform secondary analysis")
    args = parser.parse_args()

    fasta_root_folder = pathlib.Path("/mnt/data/projects/cell_atlas_approximations/reference_atlases/data/saturn/peptide_sequences/")
    embedding_root_fdn = fasta_root_folder.parent / "esm_embeddings"
    embeddings_summary_fdn = embedding_root_fdn.parent / "esm_embeddings_summaries/"
    h5ad_fdn = embeddings_summary_fdn.parent / "h5ads"
    training_output_fdn = embeddings_summary_fdn.parent / f"output_nmacro{args.n_macrogenes}_nhvg{args.n_hvg}_epochs_p{args.n_pretrain_epochs}_m{args.n_epochs}"
    if args.leaveout is not None:
        training_output_fdn = training_output_fdn.parent / f"{training_output_fdn.stem}_leaveout_{args.leaveout}"
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
            '--epochs=10',
        ]
    print(" ".join(call))
    if not args.dry:
        sp.run(" ".join(call), shell=True, check=True)

        if args.secondary_analysis:
            print("Begin secondary analysis")
            import numpy as np
            import anndata
            import scanpy as sc
            import matplotlib.pyplot as plt
            import seaborn as sns

            for result_h5ad in output_fdn.iterdir():
                if args.train and str(result_h5ad).endswith('finetuned.h5ad'):
                    break
                elif (not args.train) and str(result_h5ad).endswith('zeroshot.h5ad'):
                    break
            else:
                raise IOError("Inference output h5ad file not found")
            
            print("Load h5ad for inference and training")
            adata_inf = anndata.read_h5ad(result_h5ad)
            adata_train = anndata.read_h5ad(trained_adata_path)

            print("Limit training data to the guide/closest species")
            adata_train = adata_train[adata_train.obs["species"] == adata_inf.uns["guide_species"]]
            adata = anndata.concat([adata_inf, adata_train])

            print("Now we can make obs unique")
            adata.obs_names_make_unique()

            print("PCA")
            sc.pp.pca(adata)

            print("KNN")
            sc.pp.neighbors(adata)

            print("UMAP")
            sc.tl.umap(adata, n_components=2)

            print("Standardise some cell type names and set new column")
            def mapfun(ct):
                return {
                    "filament": "filamentous",
                    "glia": "glial",
                    "parenchyma": "parenchymal",
                }.get(ct, ct)
            adata.obs["cell_type"] = pd.Categorical(adata.obs["ref_labels"].astype(str).map(mapfun))


            print("Visualise")
            plt.ion()
            plt.close('all')

            sc.pl.umap(adata, color="species", title="Species", add_outline=True, size=20)
            fig = plt.gcf()
            fig.set_size_inches(9, 5)
            fig.tight_layout()  

            cell_types = np.sort(adata.obs["cell_type"].unique())
            colors = sns.color_palette('husl', n_colors=len(cell_types))
            palette = dict(zip(cell_types, colors))
            sc.pl.umap(adata, color="cell_type", title="Cell Type", add_outline=True, size=15, palette=dict(zip(cell_types, colors)))
            fig3 = plt.gcf()
            fig3.set_size_inches(17, 9.8)
            fig3.axes[0].legend(ncol=5, fontsize=6, bbox_to_anchor=(1,1), bbox_transform=fig3.axes[0].transAxes)
            fig3.tight_layout() 
