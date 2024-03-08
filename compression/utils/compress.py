'''
Utility functions for the compression
'''
import os
import gc
import pathlib
import numpy as np
import pandas as pd
import h5py
import scanpy as sc


def compress_tissue(
    adata_tissue,
    celltype_order,
    measurement_type="gene_expression",
):
    """Compress atlas for one tissue after data is clean, normalised, and reannotated."""
    celltypes = _get_celltype_order(
        adata_tissue.obs['cellType'].value_counts().index,
        celltype_order,
    )

    print('Compress at the cell type level')
    features = adata_tissue.var_names
    avg = pd.DataFrame(
            np.zeros((len(features), len(celltypes)), np.float32),
            index=features,
            columns=celltypes,
            )
    if measurement_type == "gene_expression":
        frac = pd.DataFrame(
                np.zeros((len(features), len(celltypes)), np.float32),
                index=features,
                columns=celltypes,
                )
    ncells = pd.Series(
            np.zeros(len(celltypes), np.int64), index=celltypes,
            )
    for celltype in celltypes:
        idx = adata_tissue.obs['cellType'] == celltype
        
        # Number of cells
        ncells[celltype] = idx.sum()

        # Average across cell type
        Xidx = adata_tissue[idx].X
        avg_ct = np.asarray(Xidx.mean(axis=0))
        # Depending if it was a matrix or not, we are already there
        if not np.isscalar(avg_ct[0]):
            avg_ct = avg_ct[0]
        avg.loc[:, celltype] = avg_ct

        if measurement_type == "gene_expression":
            frac_ct = np.asarray((Xidx > 0).mean(axis=0))
            # Depending if it was a matrix or not, we are already there
            if not np.isscalar(frac_ct[0]):
                frac_ct = frac_ct[0]
            frac_ct = frac_ct.astype(np.float32)
            frac.loc[:, celltype] = frac_ct

    # Local neighborhoods
    print('Compress at the cell state level')
    neid = _compress_neighborhoods(
        ncells,
        adata_tissue,
        measurement_type=measurement_type,
    )

    result = {
        'features': features,
        'celltype': {
            'ncells': ncells,
            'avg': avg,
            'neighborhood': neid,
        },
    }
    if measurement_type == "gene_expression":
        result['celltype']['frac'] = frac
        result['celltype']['neighborhood']['frac'] = neid['frac']

    return result


def _compress_neighborhoods(
    ncells,
    adata,
    max_cells_per_type=300,
    measurement_type='gene_expression',
    avg_neighborhoods=3,
):
    """Compress local neighborhood of a single cell type."""
    # Try something easy first, like k-means
    from sklearn.cluster import KMeans
    from scipy.spatial import ConvexHull

    features = adata.var_names

    celltypes = list(ncells.keys())

    # Subsample with some regard for cell typing
    print('   Subsampling for cell state compression')
    cell_ids = []
    for celltype, ncell in ncells.items():
        cell_ids_ct = adata.obs_names[adata.obs['cellType'] == celltype]
        if ncell > max_cells_per_type:
            idx_rand = np.random.choice(range(ncell), size=max_cells_per_type, replace=False)
            cell_ids_ct = cell_ids_ct[idx_rand]
        cell_ids.extend(list(cell_ids_ct))
    adata = adata[cell_ids].copy()
    nsub = len(cell_ids)
    print(f'   Subsampling done: {nsub} cells')

    ##############################################
    # USE AN EXISTING EMBEDDING OR MAKE A NEW ONE
    emb_keys = ['umap', 'tsne']
    for emb_key in emb_keys:
        if f'X_{emb_key}' in adata.obsm:
            break
    else:
        emb_key = 'umap'

        # Log
        sc.pp.log1p(adata)

        # Select features
        sc.pp.highly_variable_genes(adata)
        adata.raw = adata
        adata = adata[:, adata.var.highly_variable]

        # Create embedding, a proxy for cell states broadly
        sc.tl.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        points = adata.obsm[f'X_{emb_key}']

        # Back to all features for storage
        adata = adata.raw.to_adata()
        adata.obsm[f'X_{emb_key}'] = points

        # Back to cptt or equivalent for storage
        adata.X.data = np.expm1(adata.X.data)
    ##############################################

    points = adata.obsm[f'X_{emb_key}']

    # Do a global clustering, ensuring at least 3 cells
    # for each cluster so you can make convex hulls
    for n_clusters in range(avg_neighborhoods * len(celltypes), 1, -1):
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=0,
            n_init='auto',
        ).fit(points) 
        labels = kmeans.labels_

        # Book keep how many cells of each time are in each cluster
        tmp = adata.obs[['cellType']].copy()
        tmp['kmeans'] = labels
        tmp['c'] = 1.0
        ncells_per_label = (
                tmp.groupby(['kmeans', 'cellType'])
                   .size()
                   .unstack(fill_value=0)
                   .loc[:, celltypes])
        del tmp

        if ncells_per_label.sum(axis=1).min() >= 3:
            break
    else:
        raise ValueError("Cannot cluster neighborhoods")

    n_neis = kmeans.n_clusters
    nei_avg = pd.DataFrame(
            np.zeros((len(features), n_neis), np.float32),
            index=features,
            )
    nei_coords = pd.DataFrame(
            np.zeros((2, n_neis), np.float32),
            index=['x', 'y'],
            )
    convex_hulls = []
    if measurement_type == "gene_expression":
        nei_frac = pd.DataFrame(
                np.zeros((len(features), n_neis), np.float32),
                index=features,
                )
    for i in range(kmeans.n_clusters):
        idx = kmeans.labels_ == i

        # Add the average expression
        avg_i = np.asarray(adata.X[idx].mean(axis=0))
        # Depending on whether it was dense or sparse, we might already be there
        if not np.isscalar(avg_i[0]):
            avg_i = avg_i[0]
        nei_avg.iloc[:, i] = avg_i

        # Add the fraction expressing
        if measurement_type == "gene_expression":
            frac_i = np.asarray((adata.X[idx] > 0).mean(axis=0))
            # Depending on whether it was dense or sparse, we might already be there
            if not np.isscalar(frac_i[0]):
                frac_i = frac_i[0]
            frac_i = frac_i.astype(np.float32)
            nei_frac.iloc[:, i] = frac_i

        # Add the coordinates of the center
        points_i = points[idx]
        nei_coords.iloc[:, i] = points_i.mean(axis=0).astype(np.float32)

        # Add the convex hull
        hull = ConvexHull(points_i)
        convex_hulls.append(points_i[hull.vertices])

    # Clean up
    del adata
    gc.collect()

    nei_avg.columns = ncells_per_label.index
    nei_coords.columns = ncells_per_label.index
    if measurement_type == "gene_expression":
        nei_frac.columns = ncells_per_label.index

    neid = {
        'ncells': ncells_per_label,
        'avg': nei_avg,
        'coords_centroid': nei_coords,
        'convex_hull': convex_hulls,
    }
    if measurement_type == "gene_expression":
        neid['frac'] = nei_frac

    return neid


def _get_celltype_order(celltypes_unordered, celltype_order):
    '''Use global order to reorder cell types for this tissue'''
    celltypes_ordered = []
    for broad_type, celltypes_broad_type in celltype_order:
        for celltype in celltypes_broad_type:
            if celltype in celltypes_unordered:
                celltypes_ordered.append(celltype)

    celltypes_found = []
    missing_celltypes = False
    for celltype in celltypes_unordered:
        if celltype not in celltypes_ordered:
            if not missing_celltypes:
                missing_celltypes = True
                print('Missing celltypes:')
            print(celltype)
        else:
            celltypes_found.append(celltype)
    if missing_celltypes:
        print('Cell types found:')
        for celltype in celltypes_found:
            print(celltype)

    if missing_celltypes:
        raise IndexError("Missing cell types!")

    return celltypes_ordered
