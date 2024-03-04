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
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

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
    collect_store_feature_sequences,
    homogenise_features,
    )


def compute_size(fn):
    size_MB = os.stat(fn).st_size / 1024**2
    return size_MB


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
            'm_musculus',
            'm_murinus',
            'd_melanogaster',
            'x_laevis',

            # Single-organ species
            'c_hemisphaerica',
            's_pistillata',
            'a_queenslandica',
            'c_elegans',
            'd_rerio',
            'h_miamia',
            'i_pulchra',
            'l_minuta',
            'm_leidyi',
            'n_vectensis',
            's_mansoni',
            's_mediterranea',
            's_lacustris',
            't_adhaerens',
        ]

    sizes = Counter()
    for species in species_list:
        print('--------------------------------')
        print(species)
        print('--------------------------------')

        config = load_config(species)
        fn_out = output_folder / f'{species}.h5'

        # Iterate over gene expression, chromatin accessibility, etc.
        for measurement_type in config["measurement_types"]:
            config_mt = config[measurement_type]

            if "path_global" in config_mt:
                print(f"Compute size of full atlas")
                fn = config_mt["path_global"]
                sizes[(species, 'counts')] += compute_size(fn)

            else:

                # Iterate over tissues
                for itissue, tissue in enumerate(config_mt['path']):
                    print(tissue)

                    print(f"Compute size of full atlas for {tissue}")
                    fn = config_mt["path"][tissue]
                    sizes[(species, 'counts')] += compute_size(fn)

            if measurement_type == 'gene_expression':
                print('Measure size of feature sequences')
                atlas_data_folder = root_repo_folder / 'data' / 'full_atlases' / 'gene_expression' / species
                fn = atlas_data_folder / config_mt['feature_sequences']['path']
                sizes[(species, 'features')] += compute_size(fn)


        print('Measure size of compressed atlas, all inclusive')
        sizes[(species, 'output')] = compute_size(fn_out)

        sizes[(species, 'input')] = sizes[(species, 'counts')] + sizes[( species, 'features')]

    sizes = pd.Series(sizes)

    fig, ax = plt.subplots(figsize=(3.9, 2.2))
    tmp = sizes.unstack()
    xdata = tmp['input']
    ydata = tmp['output']
    ax.scatter(xdata, ydata, color='k')
    ax.set_xlabel('Size w/o approximations [MB]')
    ax.set_ylabel('Size with\napproximations [MB]')
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Equilines
    x = np.logspace(0, 5, 10)
    equilines = [1, 0.1, 0.01]
    for ratio in equilines:
        y = x * ratio
        ls = '-' if ratio == 1 else '--'
        lw = 1.5 if ratio == 1 else 1
        ax.plot(x, y, ls=ls, color='darkred', lw=lw)

    # Log-linear fit
    xdlog = np.log(xdata)
    ydlog = np.log(ydata)
    m = (xdlog @ ydlog) / (xdlog @ xdlog)
    ax.plot(x, x**m, lw=1.3, color='dodgerblue',
            label='$y = x^{'+'{:.2f}'.format(m)+'}$')
    ax.legend(loc='lower right')

    ax.grid(True)
    ax.set_ylim(3, 500)
    ax.set_xlim(3, 100000)
    ax.set_xticks([10, 100, 1000, 10000])
    fig.tight_layout()

    fig.savefig('/home/fabio/university/PI/papers/atlasapprox/draft1/figures/size_comparison.svg')

    plt.ion(); plt.show()
