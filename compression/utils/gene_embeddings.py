import numpy as np
import pandas as pd
import h5py
import hdf5plugin


def store_gene_embeddings(
    config_mt,
    species,
    fn_out,
    compression=22,
    ):
    '''Store ESM2 embeddings for gene/protein sequences.'''

    if compression:
        comp_kwargs = hdf5plugin.Zstd(clevel=compression)
    else:
        comp_kwargs = {}

    fn_in = '../data/esm_embeddings/embeddings_all.h5'
    with h5py.File(fn_in) as embs:
        features_emb = embs[species]['features'].asstr()[:]
        embeddings = embs[species]['embeddings'][:, :]
    emb_df = pd.DataFrame(embeddings, index=features_emb)

    with h5py.File(fn_out, 'r+') as h5:
        me = h5['gene_expression']

        # Fill noncoding genes with NaNs and use the same gene order as the atlas
        features = me['var_names'].asstr()[:]
        missing = list(set(features) - set(features_emb))
        missing_df = pd.DataFrame(
            np.zeros((len(missing), emb_df.shape[1]), dtype=emb_df.values.dtype),
            index=missing,
        )
        missing_df.iloc[:, :] = np.nan
        emb_df_order = pd.concat([emb_df, missing_df], axis=0).loc[features]

        egroup = me.create_dataset(
            'esm2_embedding_layer33',
            data=emb_df_order.values.astype(np.float32),
            dtype='f4',
            **comp_kwargs,
        )
