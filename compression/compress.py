# vim: fdm=indent
"""
author:     Fabio Zanini
date:       22/09/23
content:    Compress all atlases.
"""
import os
import sys
import argparse
import gc
import pickle
import pathlib
from collections import Counter
import gzip
import h5py
import numpy as np
import pandas as pd

import anndata
import scanpy as sc

from utils import (
    load_config,
    root_repo_folder,
    output_folder,
    curated_atlas_folder,
    postprocess_feature_names,
    filter_cells,
    normalise_counts,
    correct_annotations,
    compress_tissue,
    store_compressed_atlas,
    collect_store_feature_sequences,
    homogenise_features,
    store_gene_embeddings,
)


if __name__ == "__main__":

    pa = argparse.ArgumentParser()
    pa.add_argument("--species", type=str, default="")
    pa.add_argument("--maxtissues", type=int, default=1000)
    pa.add_argument(
        "--count-only", action="store_true", help="Only count cells and cell types"
    )
    args = pa.parse_args()

    count_stats = {}

    if args.species:
        species_list = args.species.split(",")
    else:
        species_list = [
            # Single-organ species
            "c_gigas",
            "c_hemisphaerica",
            "c_intestinalis",
            "s_pistillata",
            "a_queenslandica",
            "c_elegans",
            "d_rerio",
            "h_miamia",
            "h_vulgaris",
            "i_pulchra",
            "m_leidyi",
            "n_vectensis",
            "p_dumerilii",
            "p_crozieri",
            "s_mansoni",
            "s_mediterranea",
            "s_lacustris",
            "s_purpuratus",
            "t_adhaerens",
            "l_minuta",
            "a_thaliana",
            "t_aestivum",
            "z_mays",
            "f_vesca",
            "o_sativa",
            # Multi-organ species
            "h_sapiens",
            "m_musculus",
            "m_murinus",
            "d_melanogaster",
            "x_laevis",
        ]

    for ispe, species in enumerate(species_list):
        print("--------------------------------")
        print(f"{ispe + 1}. {species}")
        print("--------------------------------")

        count_stats[species] = {
            "ncells": 0,
            # "celltypes": set(),
            "celltypes_original": set(),
            "organs": set(),
            "ngenes": 0,
        }

        config = load_config(species)

        if not args.count_only:
            fn_out = output_folder / f"{species}.h5"

            # Remove existing compressed atlas file if present, but only do it at the end
            remove_old = os.path.isfile(fn_out)
            if remove_old:
                fn_out_final = fn_out
                fn_out = pathlib.Path(str(fn_out_final) + ".new")
                if fn_out.exists():
                    os.remove(fn_out)
            else:
                fn_out_final = None

        # Iterate over gene expression, chromatin accessibility, etc.
        for measurement_type in config["measurement_types"]:
            compressed_atlas = {"tissues": {}}
            config_mt = config[measurement_type]
            compressed_atlas["source"] = config_mt["source"]
            celltype_order = config_mt["cell_annotations"]["celltype_order"]

            load_params = {}
            if "load_params" in config_mt:
                load_params.update(config_mt["load_params"])

            if "path_global" in config_mt:
                print(f"Read raw atlas")
                adata = anndata.read_h5ad(config_mt["path_global"], **load_params)

            if "path_metadata_global" in config_mt:
                print("Read global metadata separately")
                meta = pd.read_csv(
                    config_mt["path_metadata_global"], sep="\t", index_col=0
                ).loc[adata.obs_names]

                if ("filter_cells_global" in config_mt) and (
                    "metadata" in config_mt["filter_cells_global"]
                ):
                    column, condition, value = config_mt["filter_cells_global"][
                        "metadata"
                    ]
                    if condition == "==":
                        meta = meta.loc[meta[column] == value]
                    elif condition == "!=":
                        meta = meta.loc[meta[column] != value]
                    elif condition == "isin":
                        meta = meta.loc[meta[column].isin(value)]
                    elif condition == "notin":
                        meta = meta.loc[~meta[column].isin(value)]
                    else:
                        raise ValueError(
                            f"Filtering condition not recognised: {condition}"
                        )

                if "tissues" in config_mt["cell_annotations"]["rename_dict"]:
                    tissues_raw = meta["tissue"].value_counts().index.tolist()
                    tdict = config_mt["cell_annotations"]["rename_dict"]["tissues"]
                    tmap = {t: tdict.get(t, t) for t in tissues_raw}
                    meta["tissue"] = meta["tissue"].map(tmap)
                    del tdict, tmap

                tissues = meta["tissue"].value_counts().index.tolist()
                tissues = sorted([t for t in tissues if t != ""])
            else:
                if "path_global" not in config_mt:
                    tissues = sorted(config_mt["path"].keys())
                else:
                    if ("filter_cells_global" in config_mt) and (
                        "metadata" in config_mt["filter_cells_global"]
                    ):
                        column, condition, value = config_mt["filter_cells_global"][
                            "metadata"
                        ]
                        if condition == "==":
                            adata = adata[adata.obs[column] == value]
                        elif condition == "!=":
                            adata = adata[adata.obs[column] != value]
                        elif condition == "isin":
                            adata = adata[adata.obs[column].isin(value)]
                        elif condition == "notin":
                            adata = adata[~adata.obs[column].isin(value)]
                        else:
                            raise ValueError(
                                f"Filtering condition not recognised: {condition}"
                            )
                    if "tissues" in config_mt:
                        tissues = config_mt["tissues"]
                    else:
                        if "tissues" in config_mt["cell_annotations"]["rename_dict"]:
                            tissues_raw = (
                                adata.obs["tissue"].value_counts().index.tolist()
                            )
                            tdict = config_mt["cell_annotations"]["rename_dict"][
                                "tissues"
                            ]
                            tmap = {t: tdict.get(t, t) for t in tissues_raw}
                            adata.obs["tissue"] = adata.obs["tissue"].map(tmap)
                            del tdict, tmap

                        tissues = adata.obs["tissue"].value_counts().index.tolist()
                        tissues = sorted([t for t in tissues if t != ""])

            tissues = tissues[: args.maxtissues]

            # Iterate over tissues
            for itissue, tissue in enumerate(tissues):
                print(tissue)
                count_stats[species]["organs"].add(tissue)

                if "path_metadata_global" in config_mt:
                    meta_tissue = meta.loc[meta["tissue"] == tissue]

                if "path_global" not in config_mt:
                    print(f"Read raw atlas for {tissue}")
                    adata_tissue = anndata.read_h5ad(
                        config_mt["path"][tissue], **load_params
                    )
                else:
                    print(f"Slice cells for {tissue}")
                    if "path_metadata_global" in config_mt:
                        adata_tissue = adata[meta_tissue.index]
                    else:
                        adata_tissue = adata[adata.obs["tissue"] == tissue]

                try:
                    if ("load_params" in config_mt) and (
                        "backed" in config_mt["load_params"]
                    ):
                        adata_tissue = adata_tissue.to_memory()

                    if "path_metadata_global" in config_mt:
                        adata_tissue.obs = meta_tissue.copy()

                    print("Postprocess feature names")
                    adata_tissue = postprocess_feature_names(adata_tissue, config_mt)

                    count_stats[species]["ncells"] += adata_tissue.n_obs

                    print("Filter cells")
                    adata_tissue = filter_cells(adata_tissue, config_mt)

                    print("Normalise")
                    adata_tissue = normalise_counts(
                        adata_tissue,
                        config_mt["normalisation"],
                        measurement_type,
                    )

                    original_annotation_column = config_mt["cell_annotations"]["column"]
                    adata_tissue.obs["cellTypeOriginal"] = pd.Categorical(
                        adata_tissue.obs[original_annotation_column]
                    )
                    count_stats[species]["celltypes_original"] |= set(
                        adata_tissue.obs["cellTypeOriginal"].cat.categories
                    )

                    if (count_stats[species]["ngenes"] == 0) and (
                        measurement_type == "gene_expression"
                    ):
                        count_stats[species]["ngenes"] = adata_tissue.n_vars

                    if not args.count_only:

                        print("Correct cell annotations")
                        adata_tissue = correct_annotations(
                            adata_tissue,
                            original_annotation_column,
                            species,
                            tissue,
                            config_mt["cell_annotations"]["rename_dict"],
                            config_mt["cell_annotations"]["require_subannotation"],
                            blacklist=config_mt["cell_annotations"]["blacklist"],
                            tissue_restricted=config_mt["cell_annotations"][
                                "tissue_restricted"
                            ],
                            subannotation_kwargs=config_mt["cell_annotations"][
                                "subannotation_kwargs"
                            ],
                        )

                        # FIXME: there is a weird  segfault with subannotations
                        # count_stats[species]["celltypes"] |= set(
                        #    adata_tissue.obs["cellType"].values
                        # )

                        print("Store curated version of full atlas, for comparison")
                        adata_tissue.write(
                            curated_atlas_folder
                            / f"{species}_{tissue}_{measurement_type}.h5ad",
                        )

                        print("Compress atlas")
                        compressed_atlas["tissues"][tissue] = compress_tissue(
                            adata_tissue,
                            celltype_order,
                        )

                finally:
                    print("Garbage collect at the end of tissue")
                    # FIXME: this is not working properly in case of exceptions
                    del adata_tissue
                    gc.collect()

            if "path_global" in config_mt:
                del adata
            if "path_metadata_global" in config_mt:
                del meta

            print("Garbage collection after tissue loop")
            gc.collect()

            if args.count_only:
                continue

            print("Homogenise feature list across organs if needed")
            homogenise_features(compressed_atlas)

            print("Garbage collection after feature homogenisation")
            gc.collect()

            # Touch disk - if anything goes wrong, remove output file
            try:
                print("Store compressed atlas")
                store_compressed_atlas(
                    fn_out,
                    compressed_atlas,
                    tissues,
                    celltype_order,
                    measurement_type=measurement_type,
                )

                print("Get features")
                features = compressed_atlas["features"]

                del compressed_atlas
                del tissues
                del celltype_order

                # Store feature sequences for gene expression only
                # TODO: we might want the ATAC peaks, but we do not
                # necessarily want to store the entire genome?
                if measurement_type == "gene_expression":
                    print("Garbage collection before storing feature sequences")
                    gc.collect()

                    print(features)

                    print("Collect and store feature sequences (peptides)")
                    collect_store_feature_sequences(
                        config_mt,
                        features,
                        measurement_type,
                        species,
                        fn_out,
                        fn_compressed_backup=fn_out_final or None,
                    )

                    # Do not store gene/protein embeddings for now, they are all together
                    # in a separate HDF5 file
                    if False:
                        print("Garbage collection before ESM/PROST embeddings")
                        gc.collect()

                        if measurement_type == "gene_expression":
                            print("Collect and store ESM/PROST embeddings")
                            store_gene_embeddings(
                                config_mt,
                                species,
                                fn_out,
                            )

                print("Garbage collection at the end of a species and measurement type")
                del features
                gc.collect()

            except:
                print("Delete corrupted partial file")
                os.remove(fn_out)
                raise

        if not args.count_only:
            if remove_old:
                print("Delete old file and rename new file to final filename")
                os.remove(fn_out_final)
                os.rename(fn_out, fn_out_final)

        else:
            print("Store count statistics")
            with open(
                f"/home/fabio/papers/atlasapprox/figures/stats/count_stats_{species}.pkl",
                "wb",
            ) as f:
                pickle.dump(count_stats[species], f)
