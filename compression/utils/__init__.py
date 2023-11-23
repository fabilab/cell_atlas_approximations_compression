'''
Utility functions for the compression
'''
import os
import gc
import pathlib
import yaml
import gzip
import numpy as np
import pandas as pd
import h5py

import scanpy as sc

from .paths import (
    root_repo_folder,
    output_folder,
)
from .config import load_config
from .preprocess import (
    filter_cells,
    normalise_counts,
    correct_annotations,
    postprocess_feature_names,
)
from .compress import (
    compress_tissue,
    store_compressed_atlas,
)
from .sequences import (
    collect_feature_sequences,
    store_compressed_feature_sequences,
)
from .feature_annotations import (
    collect_feature_annotations,
    store_compressed_feature_annotations,
)



def sanitise_gene_names(genes):
    genes_new = []
    for gene in genes:
        gene_new = gene.replace(',', ';')
        gene_new = gene_new.split(' ')[0]
        genes_new.append(gene_new)

    if len(set(genes_new)) != len(set(genes)):
        raise ValueError("Gene names are not unique after sanitisation.")

    return genes_new


