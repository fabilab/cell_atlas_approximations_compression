import gc
import numpy as np
import pandas as pd
import scanpy as sc


def postprocess_feature_names(adata, config_mt):
    """Postprocess the names of features, e.g. to match standard names on GenBank."""
    if "feature_name_postprocess" not in config_mt:
        return adata

    if "remove_prefixes" in config_mt["feature_name_postprocess"]:
        prefixes = config_mt["feature_name_postprocess"]["remove_prefixes"]
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        for prefix in prefixes:
            adata.var_names = adata.var_names.str.lstrip(prefix)

    if "substitute_final_uderscore" in config_mt["feature_name_postprocess"]:
        sub = config_mt["feature_name_postprocess"]["substitute_final_uderscore"]
        adata.var_names = adata.var_names.str.replace('_([^_]+)$', r'.\1', regex=True)

    return adata


def filter_cells(adata, config_mt):
    """Filter cells according to some parameter dictionary."""
    filter_dict = config_mt.get("filter_cells", {})

    if len(filter_dict) == 0:
        return adata

    ncells_orig = adata.shape[0]

    if "min_cells_per_type" in filter_dict:
        nmin = filter_dict["min_cells_per_type"]

        column = config_mt["cell_annotations"]["column"]
        ct_counts = adata.obs[column].value_counts()
        ct_abundant = ct_counts.index[ct_counts >= nmin]
        adata = adata[adata.obs[column].isin(ct_abundant)]

    if "unannotated" in filter_dict:
        unanno_values = filter_dict["unannotated"]
        if isinstance(unanno_values, str):
            unanno_values = [unanno_values]

        column = config_mt["cell_annotations"]["column"]
        # If a list is given, take the first one you find or fail
        if not isinstance(column, str):
            columns = column
            for column in columns:
                if column in adata.obs.columns:
                    break
            else:
                raise ValueError(
                    f"None of the cell annotation columns found: {columns}",
                )
        adata = adata[~adata.obs[column].isin(unanno_values)]

    ncells_new = adata.shape[0]

    if ncells_new < ncells_orig:
        delta = ncells_orig - ncells_new
        print(f'Filtered out {delta} cells, originally {ncells_orig} cells, {ncells_new} remaining')

    return adata


def normalise_counts(adata_tissue, input_normalisation, measurement_type="gene_expression"):
    """Normalise counts no matter what the input normalisation is."""
    if measurement_type == "gene_expression":
        if input_normalisation not in (
                "cptt", "raw", "cpm", "cpm+log", "cptt+log", "to-raw", "to-raw+cptt+log"):
            raise ValueError("Input normalisation not recognised: {input_normalisation}")

        if input_normalisation in ("to-raw", "to-raw+cptt+log"):
            adata_tissue = adata_tissue.raw.to_adata()

        if input_normalisation in ("cpm+log", "cptt+log", "to-raw+cptt+log"):
            adata_tissue.X = np.expm1(adata_tissue.X)

        if input_normalisation in ("raw", "cpm", "cpm+log", "to-raw"):
            sc.pp.normalize_total(
                adata_tissue,
                target_sum=1e4,
                key_added='coverage',
            )

        return adata_tissue

    elif measurement_type == "chromatin_accessibility":
        if input_normalisation not in ("binary", "to-binary"):
            raise ValueError("Input normalisation not recognised: {input_normalisation}")

        if input_normalisation == "to-binary":
            adata_tissue.X.data[:] = 1

        return adata_tissue

    raise ValueError("measurement type not recognised")


def subannotate(adata,
                species, annotation,
                markers,
                bad_prefixes=None,
                verbose=True,
                trash_unknown=True,
                skip_subannotation=False):
    '''This function subannotates a coarse annotation from an atlasi.

    This is ad-hoc, but that's ok for now. Examples are 'lymphocyte', which is
    a useless annotation unless you know what kind of lymphocytes these are, or
    if it's a mixed bag.
    '''
    # If skipping, return list of empty annotations - basically blacklisting
    if skip_subannotation:
        return [""] * adata.shape[0]

    if bad_prefixes is None:
        bad_prefixes = []

    markersi = markers.get(annotation, None)
    if markersi is None:
        raise ValueError(
            f'Cannot subannotate without markers for {species}, {annotation}')

    adata = adata.copy()
    sc.pp.log1p(adata)

    genes, celltypes = [], []
    for celltype, markers_ct in markersi.items():
        celltypes.append(celltype)
        for gene in markers_ct:
            if gene in adata.var_names:
                genes.append(gene)
            elif verbose:
                print('Missing gene:', gene)

    adatam = adata[:, genes].copy()

    # No need for PCA because the number of genes is small

    # Get neighbors
    sc.pp.neighbors(adatam)

    # Get communities
    sc.tl.leiden(adatam)

    adata.obs['subleiden'] = adatam.obs['leiden']
    sc.tl.rank_genes_groups(
        adata,
        'subleiden',
        method='t-test_overestim_var',
    )
    top_marker = pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(2)

    subannos = {}
    for cluster, genestop in top_marker.items():
        found = False
        for gene in genestop:
            if found:
                break
            found_bad_prefix = False
            for bad_pfx in bad_prefixes:
                if gene.startswith(bad_pfx):
                    found_bad_prefix = True
                    break
            if found_bad_prefix:
                subannos[cluster] = ''
                continue
            for celltype, markers_ct in markersi.items():
                if gene in markers_ct:
                    subannos[cluster] = celltype
                    found = True
                    break
            else:
                # FIXME: trash clusters with unknown markers for now
                if not trash_unknown:
                    import ipdb; ipdb.set_trace()
                    raise ValueError('Marker not found:', gene)
                else:
                    subannos[cluster] = ''
        if not found:
            subannos[cluster] = ''

    new_annotations = adata.obs['subleiden'].map(subannos)

    return new_annotations


def correct_annotations(
    adata,
    column,
    species,
    tissue,
    rename_dict,
    require_subannotation,
    blacklist=None,
    subannotation_kwargs=None,
):
    '''Correct cell types in each tissue according to known dict'''
    # If a list is given, take the first one you find or fail
    if not isinstance(column, str):
        columns = column
        for column in columns:
            if column in adata.obs.columns:
                break
        else:
            raise ValueError(
                f"None of the cell annotation columns found: {columns}",
            )

    # Ignore cells with NaN in the cell.type column
    idx = adata.obs[column].isin(
            adata.obs[column].value_counts().index)
    adata = adata[idx].copy()

    gc.collect()

    adata.obs[column + '_lowercase'] = adata.obs[column].str.lower()

    if blacklist is None:
        blacklist = {}
    if subannotation_kwargs is None:
        subannotation_kwargs = {}

    celltypes_new = np.asarray(adata.obs[column + '_lowercase']).copy()

    # Exclude blacklisted
    if tissue in blacklist:
        for ctraw in blacklist[tissue]:
            celltypes_new[celltypes_new == ctraw] = ''

    # Rename according to standard dict
    if 'cell_types' in rename_dict:
        for ctraw, celltype in rename_dict['cell_types'].items():
            # one can use brain:neuron for renaming in specific tissues only
            if isinstance(ctraw, str) and (':' not in ctraw):
                celltypes_new[celltypes_new == ctraw] = celltype
            else:
                # Organ-specific renames
                if isinstance(ctraw, str) and ':' in ctraw:
                    organraw, ctraw = ctraw.split(':')
                else:
                    organraw, ctraw = ctraw
                if organraw == tissue:
                    celltypes_new[celltypes_new == ctraw] = celltype

    ct_found = np.unique(celltypes_new)

    # In some data sets, some unnotated clusters are denoted by a digit
    for ctraw in ct_found:
        if ctraw.isdigit():
            celltypes_new[celltypes_new == ctraw] = ''
    ct_found = np.unique(celltypes_new)

    # Look for coarse annotations
    ctnew_list = set(celltypes_new)
    for celltype in ctnew_list:
        if celltype in require_subannotation:
            idx = celltypes_new == celltype
            adata_coarse_type = adata[idx]
            print(f'Subannotating {celltype}')
            subannotations = subannotate(
                adata_coarse_type, species, celltype,
                **subannotation_kwargs,
            )

            # Ignore reclustering into already existing types, we have enough
            for subanno in subannotations:
                if subanno in ct_found:
                    subannotations[subannotations == subanno] = ''
            print('Subannotation done')

            celltypes_new[idx] = subannotations

    adata.obs['cellType'] = celltypes_new

    # Eliminate cell types with less than 3 cells
    ncells = adata.obs['cellType'].value_counts()
    rare_celltypes = ncells.index[ncells < 3]
    adata.obs.loc[adata.obs['cellType'].isin(rare_celltypes), 'cellType'] = ''

    # Correction might declare some cells as untyped/low quality
    # they have an empty string instead of an actual annotation
    if (adata.obs['cellType'] == '').sum() > 0:
        idx = adata.obs['cellType'] != ''
        adata= adata[idx]

    return adata
