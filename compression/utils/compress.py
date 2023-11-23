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


def store_compressed_atlas(
        fn_out,
        compressed_atlas,
        tissues,
        celltype_order,
        measurement_type='gene_expression',
        compression=22,
        quantisation="chromatin_accessibility",
        #chunked=True,
        ):
    '''Store compressed atlas into h5 file.

    Args:
        fn_out: The h5 file with the compressed atlas.
        compressed_atlas: The dict with the result.
        tissues: A list of tissues covered.
        celltype_order: The order of cell types.
        measurement_type: What type of data this is (gene expression, chromatin accessibility, etc.).
        quantisation: If not None, average measurement is quantised with these bins.
        compression: Use zstd compression of the data arrays (avg and frac). Levels are 1-22,
            whereas 0 or False means no compression. No performace decrease is observed.
    '''
    add_kwargs = {}

    # Optional zstd compression using hdf5plugin
    if compression:
        import hdf5plugin
        # NOTE: decompressing zstd is equally fast no matter how much compression.
        # As for compression speed, levels 1-19 are normal, 20-22 "ultra".
        # A quick runtime test shows *faster* access for clevel=22 than clevel=3,
        # while the file size is around 10% smaller. Compression speed is significantly
        # slower, but (i) still somewhat faster than actually averaging the data and
        # (ii) compresses whole human RNA+ATAC is less than 1 minute. That's nothing
        # considering these approximations do not change that often.
        comp_kwargs = hdf5plugin.Zstd(clevel=compression)
    else:
        comp_kwargs = {}

    # Data can be quantised for further compression (typically ATAC-Seq)
    if (quantisation == True) or (quantisation == measurement_type):
        if measurement_type == "chromatin_accessibility":
            # NOTE: tried using quantiles for this, but they are really messy
            # and subject to aliasing effects. 8 bits are more than enough for
            # most biological questions given the noise in the data
            qbits = 8
            bins = np.array([-0.001, 1e-8] + np.logspace(-4, 0, 2**qbits - 1).tolist()[:-1] + [1.1])
            # bin "centers", aka data quantisation
        elif measurement_type == "gene_expression":
            # Counts per ten thousand quantisation
            qbits = 16
            bins = np.array([-0.001, 1e-8] + np.logspace(-2, 4, 2**qbits - 1).tolist()[:-1] + [1.1e4])
        else:
            raise ValueError(f"Quantisation for {measurement_type} not set.")

        quantisation_array = [0] + np.sqrt(bins[1:-2] * bins[2:-1]).tolist() + [1]

        qbytes = qbits // 8
        # Add a byte if the quantisation is not optimal
        if qbits not in (8, 16, 32, 64):
            qbytes += 1
        avg_dtype = f"u{qbytes}"
        quantisation = True
    else:
        avg_dtype = "f4"
        quantisation = False

    for tissue, group in compressed_atlas.items():
        features = group['features'].tolist()
        break

    with h5py.File(fn_out, 'a') as h5_data:
        me = h5_data.create_group(measurement_type)
        me.create_dataset('features', data=np.array(features).astype('S'))
        if quantisation:
            me.create_dataset('quantisation', data=np.array(quantisation_array).astype('f4'))

        me.create_dataset('tissues', data=np.array(tissues).astype('S'))
        supergroup = me.create_group('by_tissue')
        for tissue in tissues:
            tgroup = supergroup.create_group(tissue)
            #for label in ['celltype', 'celltype_dataset_timepoint']:
            for label in ['celltype']:
                group = tgroup.create_group(label)

                # Number of cells
                ncells = compressed_atlas[tissue][label]['ncells']
                group.create_dataset(
                    'cell_count', data=ncells.values, dtype='i8')

                # Average in a cell type
                avg = compressed_atlas[tissue][label]['avg']
                if quantisation:
                    # pd.cut wants one dimensional arrays so we ravel -> cut -> reshape
                    avg_vals = (pd.cut(avg.values.ravel(), bins=bins, labels=False)
                                .reshape(avg.shape)
                                .astype(avg_dtype))
                    avg = pd.DataFrame(
                        avg_vals, columns=avg.columns, index=avg.index,
                    )

                # TODO: manual chunking might increase performance a bit, the data is
                # typically accessed only vertically (each feature its own island)
                #if chunked:
                #    # Chunk each feature on its own: this is perfect for ATAC-Seq 
                #    add_kwargs['chunks'] = (1, len(features))

                # Cell types
                group.create_dataset(
                    'index', data=avg.columns.values.astype('S'))
                group.create_dataset(
                    'average', data=avg.T.values, dtype=avg_dtype,
                    **add_kwargs,
                    **comp_kwargs,
                )
                if measurement_type == 'gene_expression':
                    # Fraction detected in a cell type
                    frac = compressed_atlas[tissue][label]['frac']
                    group.create_dataset(
                        'fraction', data=frac.T.values, dtype='f4',
                        **add_kwargs,
                        **comp_kwargs,
                    )

                # Local neighborhoods
                neid = compressed_atlas[tissue][label]['neighborhood']
                neigroup = group.create_group('neighborhood')
                ncells = neid['ncells']
                neigroup.create_dataset(
                    'cell_count', data=ncells.values, dtype='i8')

                avg = neid['avg']
                if quantisation:
                    # pd.cut wants one dimensional arrays so we ravel -> cut -> reshape
                    avg_vals = (pd.cut(avg.values.ravel(), bins=bins, labels=False)
                                .reshape(avg.shape)
                                .astype(avg_dtype))
                    avg = pd.DataFrame(
                        avg_vals, columns=avg.columns, index=avg.index,
                    )
                neigroup.create_dataset(
                    'index', data=avg.columns.values.astype('S'))
                neigroup.create_dataset(
                    'average', data=avg.T.values, dtype=avg_dtype,
                    **add_kwargs,
                    **comp_kwargs,
                )
                if measurement_type == 'gene_expression':
                    # Fraction detected in a cell type
                    frac = neid['frac']
                    neigroup.create_dataset(
                        'fraction', data=frac.T.values, dtype='f4',
                        **add_kwargs,
                        **comp_kwargs,
                    )

                # Centroid coordinates
                coords_centroids = neid['coords_centroid']
                neigroup.create_dataset(
                    'coords_centroid',
                    data=coords_centroids.T.values, dtype=avg_dtype,
                    **add_kwargs,
                    **comp_kwargs,
                )

                # Convex hulls
                convex_hulls = neid['convex_hull']
                hullgroup = neigroup.create_group('convex_hull')
                for ih, hull in enumerate(convex_hulls):
                    hullgroup.create_dataset(
                        str(ih), data=hull, dtype='f4',
                        **add_kwargs,
                        **comp_kwargs,
                    )

        ct_group = me.create_group('celltypes')
        supertypes = np.array([x[0] for x in celltype_order])
        ct_group.create_dataset(
                'supertypes',
                data=supertypes.astype('S'),
                )
        for supertype, subtypes in celltype_order:
            ct_group.create_dataset(
                supertype,
                data=np.array(subtypes).astype('S'),
            )


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
        avg[celltype] = np.asarray(Xidx.mean(axis=0))[0]
        if measurement_type == "gene_expression":
            frac[celltype] = np.asarray((Xidx > 0).mean(axis=0))[0]

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
        nei_avg.iloc[:, i] = np.asarray(adata.X[idx].mean(axis=0))[0]
        # Add the fraction expressing
        if measurement_type == "gene_expression":
            nei_frac.iloc[:, i] = np.asarray((adata.X[idx] > 0).mean(axis=0))[0]

        # Add the coordinates of the center
        points_i = points[idx]
        nei_coords.iloc[:, i] = points_i.mean(axis=0)

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
