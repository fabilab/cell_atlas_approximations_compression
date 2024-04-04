# vim: fdm=indent
'''
author:     Fabio Zanini
date:       22/09/23
content:    Compress all atlases.
'''
import os
import sys
import argparse
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
    )



def read_features_from_h5ad(filename, **load_params):
    import h5py
    import hdf5plugin

    with h5py.File(filename, 'r') as h5_data:
        features = h5_data['var']['_index'].asstr()[:]
    return features


def postprocess_feature_names(features, config_mt):
    """Postprocess the names of features, e.g. to match standard names on GenBank."""
    if "feature_name_postprocess" not in config_mt:
        return features

    features = pd.Index(features, name='features')

    if "remove_space" in config_mt["feature_name_postprocess"]:
        features = pd.Index(
            features.str.split(" ", expand=True).get_level_values(0),
            name=features.name,
        )

    if "remove_prefixes" in config_mt["feature_name_postprocess"]:
        prefixes = config_mt["feature_name_postprocess"]["remove_prefixes"]
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        for prefix in prefixes:
            features = features.str.lstrip(prefix)

    if "substitute_final_uderscore" in config_mt["feature_name_postprocess"]:
        sub = config_mt["feature_name_postprocess"]["substitute_final_uderscore"]
        features = features.str.replace('_([^_]+)$', r'.\1', regex=True)

    return features.values


def homogenise_features(features_dict):
    """Ensure all tissues use the same features"""
    if len(features_dict['tissues']) == 1:
        for tissue, group in compressed_atlas['tissues'].items():
            return group['features']

    features_all = sorted(
        set().union(*(set(g) for g in features_dict['tissues'].values())),
    )
    features = np.asarray(features_all)
    return features


def check_feature_sequences(features, species):
    peptide_sequence_fdn = '../data/peptide_sequences/'
    fn_peptides = pathlib.Path(f'{peptide_sequence_fdn}/{species}.fasta.gz')
    if fn_peptides.exists():
        path = fn_peptides
        feature_type = 'protein'
    else:
        raise IOError(
            f'peptide file not found for species: {species}',
        )

    gene_raws = set()
    cats = {'found': 0, 'notfound': 0}
    with gzip.open(path, 'rt') as handle:
        for line in handle:
            if not line.startswith('>'):
                continue
            gene = line[1:].rstrip()
            gene_raws.add(gene)
            if gene in features:
                cats['found'] += 1
            else:
                cats['notfound'] += 1
    
    pct_nonempty = 100.0 * cats['found'] / len(features)
    print(f'Percentage of features with known sequence: {pct_nonempty}%')
    if pct_nonempty < 50:
        print('Starting ipdb debugger...')
        import ipdb; ipdb.set_trace()


if __name__ == '__main__':

    pa = argparse.ArgumentParser()
    pa.add_argument('--species', type=str, default='')
    pa.add_argument('--maxtissues', type=int, default=1000)
    args = pa.parse_args()

    if args.species:
        species_list = args.species.split(',')
    else:
        species_list = [
            # Multi-organ species
            'h_sapiens',
            #'m_musculus',
            #'m_murinus',
            #'d_melanogaster',
            #'x_laevis',

            # Single-organ species
            #'c_gigas',
            #'c_hemisphaerica',
            #'s_pistillata',
            #'a_queenslandica',
            #'c_elegans',
            #'d_rerio',
            #'h_miamia',
            #'i_pulchra',
            #'m_leidyi',
            #'n_vectensis',
            #'p_crozieri',
            #'s_mansoni',
            #'s_mediterranea',
            #'s_lacustris',
            #'s_purpuratus',
            #'t_adhaerens',

            #'l_minuta',
            #'a_thaliana',
            #'t_aestivum',
        ]

    for species in species_list:
        print('--------------------------------')
        print(species)
        print('--------------------------------')

        config = load_config(species)
        fn_out = output_folder / f'{species}.h5'

        # Remove existing compressed atlas file if present, but only do it at the end
        remove_old = os.path.isfile(fn_out)
        if remove_old:
            fn_out_final = fn_out
            fn_out = pathlib.Path(str(fn_out_final)+'.new')

        # Iterate over gene expression, chromatin accessibility, etc.
        for measurement_type in ['gene_expression']:
            features_dict = {'tissues': {}}
            config_mt = config[measurement_type]
            celltype_order = config_mt["cell_annotations"]["celltype_order"]

            load_params = {}
            if 'load_params' in config_mt:
                load_params.update(config_mt['load_params'])

            if "path_global" in config_mt:
                print(f"Read full atlas")
                features = read_features_from_h5ad(config_mt["path_global"], **load_params)

            if "path_metadata_global" in config_mt:
                print("Read global metadata separately")
                meta = pd.read_csv(config_mt["path_metadata_global"], sep='\t', index_col=0).loc[obs_names]

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

            tissues = tissues[:args.maxtissues]

            # Iterate over tissues
            for itissue, tissue in enumerate(tissues):
                print(tissue)

                if "path_global" not in config_mt:
                    print(f"Read full atlas for {tissue}")
                    features_tissue = read_features_from_h5ad(config_mt["path"][tissue], **load_params)
                else:
                    print(f'Slice cells for {tissue}')
                    features_tissue = features

                try:
                    print("Postprocess feature names")
                    features_tissue = postprocess_feature_names(features_tissue, config_mt)

                    features_dict['tissues'][tissue] = features_tissue

                finally:
                    print('Garbage collect at the end of tissue')
                    # FIXME: this is not working properly in case of exceptions
                    gc.collect()

            print('Garbage collection after tissue loop')
            gc.collect()

            print('Homogenise feature list across organs if needed')
            features = homogenise_features(features_dict)

            print('Garbage collection after feature homogenisation')
            gc.collect()

            print('Collect and store feature sequences (peptides)')
            check_feature_sequences(features, species)
