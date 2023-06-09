'''
Utility functions for the compression
'''
import os
import pathlib
import gzip
import numpy as np
import pandas as pd
import h5py

import scanpy as sc


root_repo_folder = pathlib.Path(__file__).parent.parent
# Try to put the output in the API repo if available
output_folder = root_repo_folder / '..' / 'cell_atlas_approximations_API' / 'web' / 'static' / 'atlas_data'
if not output_folder.is_dir():
    output_folder = root_repo_folder / 'data' / 'atlas_approximations'


def get_tissue_data_dict(species, atlas_folder, rename_dict=None):
    '''Get a dictionary with tissue order and files'''
    result = []

    fns = os.listdir(atlas_folder)
  
    if species == "m_fascicularis":
        fns = [x for x in fns if x.startswith('Matrix_')]
    else:
        fns = [x for x in fns if x.endswith('.h5ad')]
 
    for filename in fns:
        if species == 'mouse':
            tissue = filename.split('-')[-1].split('.')[0]
        elif species == 'human':
            tissue = filename[3:-5]
        elif species == 'lemur':
            tissue = filename[:-len('_FIRM_hvg.h5ad')].replace('_', ' ').title()
        elif species in ('c_elegans', 'd_rerio', 's_lacustris',
                         'a_queenslandica', 'm_leidyi', 't_adhaerens'):
            tissue = 'whole'
        elif species == "m_fascicularis":
            tissue = filename.split('_')[1].split('.')[0]
        else:
            raise ValueError('species not found: {:}'.format(species))

        if rename_dict is not None:
            tissue = rename_dict['tissues'].get(tissue, tissue)
            
        if species == "m_fascicularis":
            result.append({
                'tissue': tissue,
                'filename_count': filename,
                'filename_meta': filename.replace('Matrix_', 'Metadata_')[:-3],
            })
        else:      
            result.append({
                'tissue': tissue,
                'filename': atlas_folder / filename,
            })

    result = pd.DataFrame(result).set_index('tissue')
    if species != "m_fascicularis":
        result = result['filename']

    # Order tissues alphabetically
    result = result.sort_index()

    return result


def subannotate(adata,
                species, annotation,
                markers,
                bad_prefixes,
                verbose=True,
                trash_unknown=True):
    '''This function subannotates a coarse annotation from an atlasi.

    This is ad-hoc, but that's ok for now. Examples are 'lymphocyte', which is
    a useless annotation unless you know what kind of lymphocytes these are, or
    if it's a mixed bag.
    '''
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


def fix_annotations(
    adata, column, species, tissue, rename_dict, coarse_cell_types,
    blacklist=None, subannotation_kwargs=None,
):
    '''Correct cell types in each tissue according to known dict'''
    if blacklist is None:
        blacklist = {}
    if subannotation_kwargs is None:
        subannotation_kwargs = {}

    celltypes_new = np.asarray(adata.obs[column]).copy()

    # Exclude blacklisted
    if tissue in blacklist:
        for ctraw in blacklist[tissue]:
            celltypes_new[celltypes_new == ctraw] = ''

    # Rename according to standard dict
    for ctraw, celltype in rename_dict['cell_types'].items():
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
        if celltype in coarse_cell_types:
            idx = celltypes_new == celltype
            adata_coarse_type = adata[idx]
            subannotations = subannotate(
                adata_coarse_type, species, celltype,
                **subannotation_kwargs,
            )

            # Ignore reclustering into already existing types, we have enough
            for subanno in subannotations:
                if subanno in ct_found:
                    subannotations[subannotations == subanno] = ''

            celltypes_new[idx] = subannotations

    return celltypes_new


def get_celltype_order(celltypes_unordered, celltype_order):
    '''Use global order to reorder cell types for this tissue'''
    celltypes_ordered = []
    for broad_type, celltypes_broad_type in celltype_order:
        for celltype in celltypes_broad_type:
            if celltype in celltypes_unordered:
                celltypes_ordered.append(celltype)

    missing_celltypes = False
    for celltype in celltypes_unordered:
        if celltype not in celltypes_ordered:
            if not missing_celltypes:
                missing_celltypes = True
                print('Missing celltypes:')
            print(celltype)

    if missing_celltypes:
        raise IndexError("Missing cell types!")

    return celltypes_ordered


def collect_gene_annotations(anno_fn, genes):
    '''Collect gene annotations from GTF file'''
    featype = 'gene'

    with gzip.open(anno_fn, 'rt') as gtf:
        gene_annos = []
        for line in gtf:
            if f'\t{featype}\t' not in line:
                continue
            fields = line.split('\t')
            if fields[2] != featype:
                continue
            attrs = fields[-1].split(';')

            gene_name = None
            transcript_id = None
            for attr in attrs:
                if 'gene_name' in attr:
                    gene_name = attr.split(' ')[-1][1:-1]
                elif 'transcript_id' in attr:
                    transcript_id = attr.split(' ')[-1][1:-1]
                elif 'Name=' in attr:
                    gene_name = attr.split('=')[1]
                    transcript_id = gene_name

            if (gene_name is not None) and (transcript_id is None):
                transcript_id = gene_name

            if (gene_name is None) or (transcript_id is None):
                continue
            gene_annos.append({
                'transcript_id': transcript_id,
                'gene_name': gene_name,
                'chromosome_name': fields[0],
                'start_position': int(fields[3]),
                'end_position': int(fields[4]),
                'strand': 1 if fields[6] == '+' else -1,
                'transcription_start_site': int(fields[3]) if fields[6] == '+' else int(fields[4]),
                })
    gene_annos = pd.DataFrame(gene_annos)

    # NOTE: some species like zebrafish don't really have a transcript id (yet?)
    #assert gene_annos['transcript_id'].value_counts()[0] == 1

    # FIXME: choose the largest transcript or something. For this particular
    # repo it's not that important
    gene_annos = (gene_annos.drop_duplicates('gene_name')
                            .set_index('gene_name', drop=False))

    genes_missing = list(set(genes) - set(gene_annos['gene_name'].values))
    gene_annos_miss = pd.DataFrame([], index=genes_missing)
    gene_annos_miss['transcript_id'] = gene_annos_miss.index
    gene_annos_miss['start_position'] = -1
    gene_annos_miss['end_position'] = -1
    gene_annos_miss['strand'] = 0
    gene_annos_miss['chromosome_name'] = ''
    gene_annos_miss['transcription_start_site'] = -1
    gene_annos = pd.concat([gene_annos, gene_annos_miss])
    gene_annos = gene_annos.loc[genes]
    gene_annos['strand'] = gene_annos['strand'].astype('i2')

    return gene_annos


def store_compressed_atlas(
        fn_out,
        compressed_atlas,
        tissues,
        feature_annos,
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
        feature_annos: Gene annotations if available (only for gene expression).
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

        quantisation = [0] + np.sqrt(bins[1:-2] * bins[2:-1]).tolist() + [1]

        qbytes = qbits // 8
        # Add a byte if the quantisation is not optimal
        if qbits not in (8, 16, 32, 64):
            qbytes += 1
        avg_dtype = f"u{qbytes}"
        quantisation = True
    else:
        avg_dtype = "f4"
        quantisation = False

    if feature_annos is not None:
        features = feature_annos.index.tolist()
    else:
        for tissue, group in compressed_atlas.items():
            features = group['features'].tolist()

    with h5py.File(fn_out, 'a') as h5_data:
        me = h5_data.create_group(measurement_type)
        me.create_dataset('features', data=np.array(features).astype('S'))
        if quantisation:
            me.create_dataset('quantisation', data=np.array(quantisation).astype('f4'))

        if feature_annos is not None:
            group = me.create_group('feature_annotations')
            group.create_dataset(
                    'gene_name', data=feature_annos.index.values.astype('S'))
            group.create_dataset(
                    'transcription_start_site',
                    data=feature_annos['transcription_start_site'].values, dtype='i8')
            group.create_dataset(
                    'chromosome_name',
                    data=feature_annos['chromosome_name'].astype('S'))
            group.create_dataset(
                    'start_position',
                    data=feature_annos['start_position'].values, dtype='i8')
            group.create_dataset(
                    'end_position',
                    data=feature_annos['end_position'].values, dtype='i8')
            group.create_dataset(
                    'strand', data=feature_annos['strand'].values, dtype='i2')

        me.create_dataset('tissues', data=np.array(tissues).astype('S'))
        supergroup = me.create_group('by_tissue')
        for tissue in tissues:
            tgroup = supergroup.create_group(tissue)
            #for label in ['celltype', 'celltype_dataset_timepoint']:
            for label in ['celltype']:
                ncells = compressed_atlas[tissue][label]['ncells']
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

                group = tgroup.create_group(label)
                # Cell types
                group.create_dataset(
                        'index', data=avg.columns.values.astype('S'))
                group.create_dataset(
                    'average', data=avg.T.values, dtype=avg_dtype,
                    **add_kwargs,
                    **comp_kwargs,
                )
                group.create_dataset(
                    'cell_count', data=ncells.values, dtype='i8')

                if measurement_type == 'gene_expression':
                    frac = compressed_atlas[tissue][label]['frac']
                    group.create_dataset(
                        'fraction', data=frac.T.values, dtype='f4',
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
