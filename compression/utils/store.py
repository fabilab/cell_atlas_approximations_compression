'''
Utility functions for storage.
'''
import os
import gc
import pathlib
import h5py
import numpy as np
import pandas as pd


atlasapprox_compression_version = 'v1.2'



class AtlasApproxStorer():
    def __init__(
        self,
        quantisation,
        avg_dtype,
        add_kwargs,
        comp_kwargs,
        measurement_type,
    ):
        self.quantisation = quantisation
        self.avg_dtype = avg_dtype
        self.add_kwargs = add_kwargs
        self.comp_kwargs = comp_kwargs
        self.measurement_type = measurement_type

    def store_cell_types(self, comp_data, group):
        """Store compressed atlas at cell type information level"""
        # Cell type names
        celltypes_tissue = comp_data['avg'].columns.values
        group.create_dataset(
            'obs_names', data=celltypes_tissue.astype('S'))
    
        # Number of cells
        ncells = comp_data['ncells']
        group.create_dataset(
            'cell_count', data=ncells.values, dtype='i8')
    
        # Average expression
        avg = comp_data['avg']
        if self.quantisation:
            # pd.cut wants one dimensional arrays so we ravel -> cut -> reshape
            avg_vals = (pd.cut(avg.values.ravel(), bins=bins, labels=False)
                        .reshape(avg.shape)
                        .astype(self.avg_dtype))
            avg = pd.DataFrame(
                avg_vals, columns=avg.columns, index=avg.index,
            )
        group.create_dataset(
            'average', data=avg.T.values, dtype=self.avg_dtype,
            **self.add_kwargs,
            **self.comp_kwargs,
        )
    
        # Fraction of cells with detected molecules
        if self.measurement_type == 'gene_expression':
            frac = comp_data['frac']
            group.create_dataset(
                'fraction', data=frac.T.values, dtype='f4',
                **self.add_kwargs,
                **self.comp_kwargs,
            )
    
    
    def store_cell_states(self, neid, neigroup):
        """Store compressed atlas at cell state information level"""
    
        # Number of cells
        ncells = neid['ncells']
        neigroup.create_dataset(
            'cell_count', data=ncells.values, dtype='i8')
    
        # Average expression
        avg = neid['avg']
        if self.quantisation:
            # pd.cut wants one dimensional arrays so we ravel -> cut -> reshape
            avg_vals = (pd.cut(avg.values.ravel(), bins=bins, labels=False)
                        .reshape(avg.shape)
                        .astype(self.avg_dtype))
            avg = pd.DataFrame(
                avg_vals, columns=avg.columns, index=avg.index,
            )
        neigroup.create_dataset(
            'average', data=avg.T.values, dtype=self.avg_dtype,
            **self.add_kwargs,
            **self.comp_kwargs,
        )
    
        # Cell state "names"
        neigroup.create_dataset(
            'obs_names', data=avg.columns.values.astype('S'))
        
        # Fraction of cells with detected molecules
        if self.measurement_type == 'gene_expression':
            frac = neid['frac']
            neigroup.create_dataset(
                'fraction', data=frac.T.values, dtype='f4',
                **self.add_kwargs,
                **self.comp_kwargs,
            )
    
        # Cell state centroid coordinates
        coords_centroids = neid['coords_centroid']
        neigroup.create_dataset(
            'coords_centroid',
            data=coords_centroids.T.values, dtype='f4',
            **self.add_kwargs,
            **self.comp_kwargs,
        )
    
        # Cell state convex hulls
        convex_hulls = neid['convex_hull']
        hullgroup = neigroup.create_group('convex_hull')
        for ih, hull in enumerate(convex_hulls):
            hullgroup.create_dataset(
                str(ih), data=hull, dtype='f4',
                **self.add_kwargs,
                **self.comp_kwargs,
            )


def store_compressed_atlas(
        fn_out,
        compressed_atlas_with_features,
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
        compressed_atlas_with_features: The dict with the result.
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

    storer = AtlasApproxStorer(
        quantisation=quantisation,
        avg_dtype=avg_dtype,
        add_kwargs=add_kwargs,
        comp_kwargs=comp_kwargs,
        measurement_type=measurement_type,
    )

    with h5py.File(fn_out, 'a') as h5_data:
        if 'atlasapprox_compression_version' not in h5_data.attrs:
            h5_data.attrs['software'] = 'atlasapprox_compression'
            h5_data.attrs['version'] = atlasapprox_compression_version

        if 'measurements' not in h5_data:
            megroup = h5_data.create_group('measurements')
        else:
            megroup = h5_data['measurements']

        me = megroup.create_group(measurement_type)

        # Store feature sequences
        features = list(compressed_atlas_with_features['features'])
        me.create_dataset('var_names', data=np.array(features).astype('S'))
    
        compressed_atlas = compressed_atlas_with_features['tissues']        

        if quantisation:
            me.create_dataset('quantisation', data=np.array(quantisation_array).astype('f4'))

        # Extract organism-wide cell types, in a specific order to help visualisation
        # NOTE: we do not need to keep track of "supertypes" since they are sloppily defined as of now
        celltypes = []
        for supertype, subtypes in celltype_order:
            celltypes.extend(list(subtypes))

        # Specify how the data is grouped
        gby_group = me.create_group('grouped_by')
        data_group = me.create_group('data')

        # This could be generalised a bit, but it's ok for now
        glabels = ['tissue->celltype']
        gby_values = [tissues, celltypes]
        gby_types = ['S', 'S']
        for glabel in glabels:
            gby_levels = glabel.split('->')

            gby_groupgl = gby_group.create_group(glabel)
            data_groupgl = data_group.create_group(glabel)

            gby_groupgl.create_dataset('names', data=np.array(gby_levels).astype('S'))
            gby_groupgl.create_dataset('dtypes', data=np.array(['object', 'object']).astype('S'))
            valgroup = gby_groupgl.create_group('values')
            for gby_level, gby_value, gby_type in zip(gby_levels, gby_values, gby_types):
                valgroup.create_dataset(gby_level, data=np.array(gby_value).astype(gby_type))

            # NOTE: separating by tissue is useful for the tissue-specific UMAPs. We can think
            # further about how to combine that with organism-wide UMAPs and cell states.
            for tissue in gby_values[0]:
                group = data_groupgl.create_group(tissue)
                comp_data = compressed_atlas[tissue][gby_levels[1]]

                ############################################
                # Biology-driven compression (cell types)
                ############################################
                storer.store_cell_types(comp_data, group)

                ############################################
                # Data-driven compression (cell states)
                # NOTE: Cell states are tissue-specific
                ############################################
                neigroup = group.create_group('neighborhood')
                storer.store_cell_states(comp_data['neighborhood'], neigroup)

