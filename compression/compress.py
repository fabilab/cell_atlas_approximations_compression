# vim: fdm=indent
'''
author:     Fabio Zanini
date:       22/09/23
content:    Compress isodiametra pulchra, an acoel.
'''
import os
import sys
import gc
import pathlib
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
    postprocess_feature_names,
    filter_cells,
    normalise_counts,
    correct_annotations,
    compress_tissue,
    store_compressed_atlas,
    collect_feature_sequences,
    store_compressed_feature_sequences,
    collect_feature_annotations,
    )


if __name__ == '__main__':

    species_list = [
        # Multi-organ species
        #'h_sapiens',
        #'m_musculus',
        #'m_murinus',
        'd_melanogaster',
        'x_laevis',

        ## Single-organ species
        #'c_hemisphaerica',
        #'s_pistillata',
        #'a_queenslandica',
        #'c_elegans',
        #'d_rerio',
        #'h_miamia',
        #'i_pulchra',
        #'l_minuta',
        #'m_leidyi',
        #'n_vectensis',
        #'s_mansoni',
        #'s_mediterranea',
        #'s_lacustris',
        #'t_adhaerens',
    ]

    for species in species_list:
        print('--------------------------------')
        print(species)
        print('--------------------------------')

        config = load_config(species)
        atlas_data_folder = root_repo_folder / 'data' / 'full_atlases' / 'RNA' / species
        fn_out = output_folder / f'{species}.h5'

        # Remove existing compressed atlas file if present, but only do it at the end
        remove_old = os.path.isfile(fn_out)
        if remove_old:
            fn_out_final = fn_out
            fn_out = pathlib.Path(str(fn_out_final)+'.new')

        # Iterate over gene expression, chromatin accessibility, etc.
        for measurement_type in config["measurement_types"]:
            compressed_atlas = {}
            config_mt = config[measurement_type]
            celltype_order = config_mt["cell_annotations"]["celltype_order"]

            load_params = {}
            if 'load_params' in config_mt:
                load_params.update(config_mt['load_params'])

            if "path_global" in config_mt:
                print(f"Read full atlas")
                adata = anndata.read(config_mt["path_global"], **load_params)

            if "path_metadata_global" in config_mt:
                print("Read global metadata separately")
                meta = pd.read_csv(config_mt["path_metadata_global"], sep='\t', index_col=0).loc[adata.obs_names]

                if ("filter_cells_global" in config_mt) and ("metadata" in config_mt["filter_cells_global"]):
                    column, condition, value = config_mt["filter_cells_global"]["metadata"]
                    if condition == '==':
                        meta = meta.loc[meta[column] == value]
                    elif condition == '!=':
                        meta = meta.loc[meta[column] != value]
                    elif condition == 'isin':
                        meta = meta.loc[meta[column].isin(value)]
                    elif condition == 'notin':
                        meta = meta.loc[~meta[column].isin(value)]
                    else:
                        raise ValueError(f'Filtering condition not recognised: {condition}')

                if 'tissues' in config_mt['cell_annotations']['rename_dict']:
                    tissues_raw = meta['tissue'].value_counts().index.tolist()
                    tdict = config_mt['cell_annotations']['rename_dict']['tissues']
                    tmap = {t: tdict.get(t, t) for t in tissues_raw}
                    meta['tissue'] = meta['tissue'].map(tmap)
                    del tdict, tmap

                tissues = meta['tissue'].value_counts().index.tolist()
                tissues = sorted([t for t in tissues if t != ''])
            else:
                if "path_global" not in config_mt:
                    tissues = sorted(config_mt["path"].keys())
                else:
                    if ("filter_cells_global" in config_mt) and ("metadata" in config_mt["filter_cells_global"]):
                        column, condition, value = config_mt["filter_cells_global"]["metadata"]
                        if condition == '==':
                            adata = adata[adata.obs[column] == value]
                        elif condition == '!=':
                            adata = adata[adata.obs[column] != value]
                        elif condition == 'isin':
                            adata = adata[adata.obs[column].isin(value)]
                        elif condition == 'notin':
                            adata = adata[~adata.obs[column].isin(value)]
                        else:
                            raise ValueError(f'Filtering condition not recognised: {condition}')
                    if 'tissues' in config_mt:
                        tissues = config_mt["tissues"]
                    else:
                        if 'tissues' in config_mt['cell_annotations']['rename_dict']:
                            tissues_raw = adata.obs['tissue'].value_counts().index.tolist()
                            tdict = config_mt['cell_annotations']['rename_dict']['tissues']
                            tmap = {t: tdict.get(t, t) for t in tissues_raw}
                            adata.obs['tissue'] = adata.obs['tissue'].map(tmap)
                            del tdict, tmap

                        tissues = adata.obs['tissue'].value_counts().index.tolist()
                        tissues = sorted([t for t in tissues if t != ''])

            # Iterate over tissues
            for tissue in tissues:
                print(tissue)

                if "path_metadata_global" in config_mt:
                    meta_tissue = meta.loc[meta['tissue'] == tissue]

                if "path_global" not in config_mt:
                    print(f"Read full atlas for {tissue}")
                    adata_tissue = anndata.read(config_mt["path"][tissue], **load_params)
                else:
                    print(f'Slice cells for {tissue}')
                    if "path_metadata_global" in config_mt:
                        adata_tissue = adata[meta_tissue.index]
                    else:
                        adata_tissue = adata[adata.obs['tissue'] == tissue]

                try:
                    if ("load_params" in config_mt) and ("backed" in config_mt["load_params"]):
                        adata_tissue = adata_tissue.to_memory()
                    
                    if "path_metadata_global" in config_mt:
                        adata_tissue.obs = meta_tissue.copy()

                    print("Postprocess feature names")
                    adata_tissue = postprocess_feature_names(adata_tissue, config_mt)

                    print("Filter cells")
                    adata_tissue = filter_cells(adata_tissue, config_mt)

                    print("Normalise")
                    adata_tissue = normalise_counts(
                        adata_tissue,
                        config_mt['normalisation'],
                        measurement_type,
                    )

                    print("Correct cell annotations")
                    adata_tissue = correct_annotations(
                        adata_tissue,
                        config_mt['cell_annotations']['column'],
                        species,
                        tissue,
                        config_mt['cell_annotations']['rename_dict'],
                        config_mt['cell_annotations']['require_subannotation'],
                        blacklist=config_mt['cell_annotations']['blacklist'],
                        subannotation_kwargs=config_mt['cell_annotations']['subannotation_kwargs'],
                    )

                    print("Compress atlas")
                    compressed_atlas[tissue] = compress_tissue(
                        adata_tissue, celltype_order,
                    )

                    print('Get features')
                    # TODO: harmonise across tissues, sometimes (e.g. fly) that is not a given
                    features = adata_tissue.var_names

                finally:
                    print('Garbage collect at the end of tissue')
                    # FIXME: this is not working properly in case of exceptions
                    del adata_tissue
                    gc.collect()

            if "path_global" in config_mt:
                del adata
            if "path_metadata_global" in config_mt:
                del meta

            print('Garbage collection before storing compressed atlas')
            gc.collect()

            print('Store compressed atlas')
            store_compressed_atlas(
                    fn_out,
                    compressed_atlas,
                    tissues,
                    celltype_order,
                    measurement_type=measurement_type,
            )

            del compressed_atlas
            del tissues
            del celltype_order

            if "feature_sequences" in config_mt:
                print('Garbage collection before storing feature sequences')
                gc.collect()

                print('Collect feature sequences')
                feature_sequences = collect_feature_sequences(
                    config_mt,
                    features,
                    measurement_type, species,
                )

                print('Store feature sequences')
                store_compressed_feature_sequences(
                    fn_out,
                    feature_sequences,
                    measurement_type,
                )

                del feature_sequences

            # FIXME
            if False:
                print('Garbage collection before storing feature annotations')
                gc.collect()

                print('Collect feature annotations')
                feature_annos = collect_feature_annotations(
                        config_mt['feature_annotation'],
                        features,
                        measurement_type,
                )

                if feature_annos is not None:
                    print('Store feature annotations')
                    store_compressed_feature_annotations(
                        fn_out,
                        feature_annos,
                        measurement_type,
                    )

                del feature_annos

            print('Garbage collection at the end of a species and measurement type')
            gc.collect()

        if remove_old:
            print('Delete old file and rename new file to final filename')
            os.remove(fn_out_final)
            os.rename(fn_out, fn_out_final)
