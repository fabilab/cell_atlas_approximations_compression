"""Prepare FASTA files with peptides from all species for ESM embedding."""
import os
import pathlib
import anndata
import scquill
import glob
import argparse


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
        idx = df[df[colname] == ct].sample(n_cells_per_type, replace=False).index
        df.loc[idx, "pick"] = True

    idx = df.index[df["pick"]]

    return adata[idx]



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prepare FASTA files with peptides from all species for ESM embedding.")
    parser.add_argument('--species', default=None, help="Only process these species")
    args = parser.parse_args()

    approx_fdn = pathlib.Path("/home/fabio/projects/compressed_atlas/cell_atlas_approximations_compression/data/atlas_approximations")
    curated_fdn = pathlib.Path("/mnt/data/projects/cell_atlas_approximations/reference_atlases/data/curated_atlases/")
    output_fdn = curated_fdn.parent / "saturn"
    output_fasta_fdn = output_fdn / "peptide_sequences"
    output_h5ads_fdn = output_fdn / "h5ads"
    os.makedirs(output_fdn, exist_ok=True)
    os.makedirs(output_fasta_fdn, exist_ok=True)
    os.makedirs(output_h5ads_fdn, exist_ok=True)

    print("Build species dictionary")
    species_dict = {}
    for approx_fn in approx_fdn.iterdir():
        species = approx_fn.stem
        if args.species is not None and species not in args.species:
            continue
        print(species)
        
        curated_fns_species = glob.glob(str(curated_fdn / f"{species}*_gene_expression.h5ad"))
        if len(curated_fns_species) == 0:
            print(f"No curated atlas found for {species}")
            continue

        species_dict[species] = {
            "approximation": approx_fn,
            "curated_atlases": list(map(pathlib.Path, curated_fns_species))
        }


    print("Extract peptide sequences and merge h5ads")
    for species, datum in species_dict.items():
        print(species)
        approx = scquill.Approximation.read_h5(datum["approximation"])
        approx_adata = approx.to_anndata(measurement_type="gene_expression")
        features = approx_adata.var_names
        sequences = scquill.utils.get_feature_sequences(approx, measurement_type="gene_expression")
        if (species == "a_thaliana"):
            for i, sequence in enumerate(sequences):
                if '*' in sequence:
                    sequences[i] = sequence[:sequence.index('*')]

        idx_has_sequence = sequences != ""
        features = features[idx_has_sequence]
        sequences = sequences[idx_has_sequence]
        nseq = len(sequences)
        print(f"  {nseq} features with sequences")

        if nseq < 1000:
            print("  Less than 1000 features with peptide sequence, skipping.")
            continue

        print("  Write FASTA file with peptides")
        fasta_fn = output_fasta_fdn / f"{species}.fasta"
        with open(fasta_fn, "w") as fasta_f:
            for feature, sequence in zip(features, sequences):
                fasta_f.write(f">{feature}\n")
                fasta_f.write(f"{sequence}\n")

        print("  Merge curated atlases")
        adatas = []
        tissues = []
        for h5ad_fn in datum["curated_atlases"]:
            tissue = h5ad_fn.stem[len(species) + 1: -len("_gene_expression")]
            print(f"    {tissue}")
            adata = anndata.read_h5ad(h5ad_fn)
            adata.obs["organ"] = tissue

            # Select only some columns to avoid weird errors with scrublet etc. (fly data, of course)
            adata.obs = adata.obs[["organ", "cellType"]]

            print("      Subsample respecting cell types")
            adata_sub = subsample_adata_by_cell_type(adata)
            n_cell_types = len(adata.obs["cellType"].value_counts())
            print(f"     Originally {adata.n_obs} cells across {n_cell_types} cell types: subsampled {adata_sub.n_obs} cells")

            tissues.append(tissue)
            adatas.append(adata_sub)

        adata = anndata.concat(adatas, join="outer", index_unique="-", fill_value=0, keys=tissues)
        adata.obs_names = [f"{species}-{x}" for x in adata.obs_names]

        print("  Restrict merged adata to features with peptide sequences")
        adata = adata[:, features]


        print("  Store to h5ad file")
        adata.write_h5ad(output_h5ads_fdn / f"{species}.h5ad")
