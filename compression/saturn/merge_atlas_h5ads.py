"""Prepare h5ads from all species for SATURN, merging by species."""

import os
import pathlib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix
import anndata
import glob
import argparse
import torch


def subsample_adata_by_cell_type(
    adata,
    colname="cellType",
    n_cells_per_type=200,
):
    """Subsample anndata object by cell type."""
    df = adata.obs[[colname]].copy()
    df["pick"] = False
    for ct, numct in df[colname].value_counts().items():
        if numct <= n_cells_per_type:
            df.loc[df[colname] == ct, "pick"] = True
            continue
        # This is where the RNG happens, using pandas
        idx = df[df[colname] == ct].sample(n_cells_per_type, replace=False).index
        df.loc[idx, "pick"] = True

    idx = df.index[df["pick"]]

    return adata[idx]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Prepare FASTA files with peptides from all species for ESM embedding."
    )
    parser.add_argument("--species", default=None, help="Only process these species")
    args = parser.parse_args()

    approx_fdn = pathlib.Path(
        "/home/fabio/projects/compressed_atlas/cell_atlas_approximations_compression/data/atlas_approximations"
    )
    curated_fdn = pathlib.Path(
        "/mnt/data/projects/cell_atlas_approximations/reference_atlases/data/curated_atlases/"
    )

    saturn_data_fdn = curated_fdn.parent / "saturn"
    embedding_summary_fdn = saturn_data_fdn / "esm_embeddings_summaries/"
    output_h5ads_fdn = saturn_data_fdn / "h5ads"
    os.makedirs(output_h5ads_fdn, exist_ok=True)

    print("Build species dictionary")
    species_dict = {}
    for approx_fn in approx_fdn.iterdir():
        species = approx_fn.stem
        if args.species is not None and species not in args.species:
            continue
        print(species)

        curated_fns_species = glob.glob(
            str(curated_fdn / f"{species}*_gene_expression.h5ad")
        )
        if len(curated_fns_species) == 0:
            print(f"No curated atlas found for {species}")
            continue

        embedding_summary_fn = embedding_summary_fdn / f"{species}_gene_all_esm1b.pt"
        if not embedding_summary_fn.exists():
            print(f"No embedding summary found for {species}")

        species_dict[species] = {
            "approximation": approx_fn,
            "curated_atlases": list(map(pathlib.Path, curated_fns_species)),
            "embedding_summary": embedding_summary_fn,
        }

    print("See if the trained SATURN model has a certain subsampling already")
    print(
        "That subsample would have the same strategy as here below, but a different random seed, and we want to keep it to save on model training runs only."
    )
    model_fn = (
        saturn_data_fdn
        / "output_nmacro1000_nhvg2000_epochs_p100_m50"
        / "saturn_results"
        / "final_adata.h5ad"
    )
    if model_fn.exists():
        print("Found SATURN model, will use it to subsample the curated atlases.")
        model_obs_names = anndata.read_h5ad(model_fn).obs_names
    else:
        print(
            "SATURN model NOT found, will subsample the curated atlases from scratch."
        )
        model_obs_names = None

    print("Extract peptide embeddings and merge h5ads")
    for species, datum in species_dict.items():
        print(species)
        if model_obs_names is not None:
            model_obs_names_species = [
                x for x in model_obs_names if x.startswith(species)
            ]

        embedding_summary = torch.load(datum["embedding_summary"])

        nemb = len(embedding_summary)
        print(f"  {nemb} features with embeddings")

        if nemb < 1000:
            print("  Less than 1000 features with peptide embeddings, skipping.")
            continue

        features = list(embedding_summary.keys())

        print("  Merge curated atlases")
        adatas = []
        tissues = []
        for h5ad_fn in datum["curated_atlases"]:
            tissue = h5ad_fn.stem[len(species) + 1 : -len("_gene_expression")]
            print(f"    {tissue}")
            adata = anndata.read_h5ad(h5ad_fn)
            adata.obs["organ"] = tissue

            print("    Reverting to raw counts, to make the ZINB model happy")
            if "n_counts" not in adata.obs.columns:
                for col in ["coverage", "nCount_RNA"]:
                    if col in adata.obs.columns:
                        adata.obs["n_counts"] = adata.obs[col]
                        break
                else:
                    print(adata.obs)
                    raise ValueError("Missing column for normalisation!")

            # Check if sparse or dense, and recast as sparse int32 anyway
            if isinstance(adata.X, np.ndarray):
                adata.X = (adata.X.T * 1e-4 * adata.obs["n_counts"].values).T
                adata.X = adata.X.round(0).astype(np.int32)
                adata.X = csr_matrix(adata.X)
            else:
                if not isinstance(adata.X, csr_matrix):
                    adata.X = adata.X.tocsr()
                for i in range(adata.n_obs):
                    i0, i1 = adata.X.indptr[i : i + 2]
                    adata.X.data[i0:i1] *= 1e-4 * adata.obs["n_counts"].values[i]
                adata.X.data = adata.X.data.round(0).astype(np.int32)

            # Select only some columns to avoid weird errors with scrublet etc. (fly data, of course)
            adata.obs = adata.obs[["organ", "cellType"]]

            print("      Subsample balancing cell types")
            if model_obs_names is not None:
                model_obs_names_tissue = [
                    x[len(species) + 1 : -1 - len(tissue)]
                    for x in model_obs_names_species
                ]
                cell_id_overlap = [
                    x for x in adata.obs_names if x in model_obs_names_tissue
                ]
                if len(cell_id_overlap) == 0:
                    print(
                        f"Cells from trained SATURN model not found for {species} {tissue}"
                    )
                    __import__("ipdb").set_trace()
                adata_sub = adata[cell_id_overlap]
            else:
                adata_sub = subsample_adata_by_cell_type(adata)
            n_cell_types = len(adata.obs["cellType"].value_counts())
            print(
                f"     Originally {adata.n_obs} cells across {n_cell_types} cell types: subsampled {adata_sub.n_obs} cells"
            )

            tissues.append(tissue)
            adatas.append(adata_sub)

        adata = anndata.concat(
            adatas, join="outer", index_unique="-", fill_value=0, keys=tissues
        )
        adata.obs_names = [f"{species}-{x}" for x in adata.obs_names]

        if model_obs_names is not None:
            adata = adata[model_obs_names_species]

        print("  Restrict merged adata to features with peptide sequences")
        adata = adata[:, features]

        print("  Store to h5ad file")
        adata.write_h5ad(output_h5ads_fdn / f"{species}.h5ad")
