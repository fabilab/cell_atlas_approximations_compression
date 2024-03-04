import re
import gzip
import h5py
import numpy as np
import pandas as pd

from .paths import root_repo_folder


def collect_store_feature_sequences(
    config_mt,
    features,
    measurement_type,
    species,
    fn_out,
    compression=22, 
    ):
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

    algo = config_mt['feature_sequences'].get('algorithm', 'bulk')
    if algo == 'bulk':
        fun = _collect_store_feature_sequences_bulk
    else:
        fun = _collect_store_feature_sequences_variable

    return fun(
        config_mt,
        features,
        measurement_type,
        species,
        fn_out,
        comp_kwargs=comp_kwargs,
    )


def _collect_store_feature_sequences_bulk(
    config_mt,
    features,
    measurement_type,
    species,
    fn_out,
    comp_kwargs,
    ):
    """Collect sequences of features and store to file, all at once (better comp)."""

    # 1. Collect sequences
    path = config_mt['feature_sequences']['path']
    path = root_repo_folder / 'data' / 'full_atlases' / measurement_type / species / path
    seqs = {fea: '' for fea in features}
    with gzip.open(path, 'rt') as f:

        ## FIXME
        #missing = []

        for gene, seq in _SimpleFastaParser(f):
            # Sometimes they need a gene/id combo from biomart
            # Do this only if no finer regex is set.
            if "replace" not in config_mt["feature_sequences"]:
                if '|' in gene:
                    gene = gene.split('|')[0]
                if ' ' in gene:
                    gene = gene.split()[0]
            else:
                pattern = config_mt["feature_sequences"]["replace"]["in"]
                repl = config_mt["feature_sequences"]["replace"]["out"]
                gene = re.sub(pattern, repl, gene)

            if gene == '':
                continue
            if gene in features:
                seqs[gene] = seq
            #else:
            #    gene2 = gene.split('|')[1]
            #    missing.append(gene2)

    
    #features2 = features.str.split('|', expand=True).get_level_values(1)
    #import ipdb; ipdb.set_trace()

    seqs = pd.Series(seqs).loc[features]

    # 2. Store sequences
    with h5py.File(fn_out, 'a') as h5_data:
        me = h5_data[measurement_type]
        group = me.create_group('feature_sequences')
        group.attrs["type"] = config_mt['feature_sequences']['type']

        # Bulk strings seem to be compresed better
        group.create_dataset(
            "sequences", data=seqs.values.astype('S'),
            **comp_kwargs,
        )


def _collect_store_feature_sequences_variable(
    config_mt,
    features,
    measurement_type,
    species,
    fn_out,
    comp_kwargs,
    ):
    """Collect sequences of features and store to file (less RAM)."""

    path = config_mt['feature_sequences']['path']
    path = root_repo_folder / 'data' / 'full_atlases' / measurement_type / species / path
    featuress = pd.Series(np.arange(len(features)), index=features)

    with h5py.File(fn_out, 'a') as h5_data, gzip.open(path, 'rt') as f:
        me = h5_data[measurement_type]
        group = me.create_group('feature_sequences')
        group.attrs["type"] = config_mt['feature_sequences']['type']

        # Variable length strings... seems like it's not compressed very well or at all
        seqs = group.create_dataset(
            'sequences',
            shape=len(features),
            dtype=h5py.string_dtype(),
            **comp_kwargs,
        )

        for gene, seq in _SimpleFastaParser(f):
            # Sometimes they need a gene/id combo from biomart
            # Do this only if no finer regex is set.
            if "replace" not in config_mt["feature_sequences"]:
                if '|' in gene:
                    gene = gene.split('|')[0]
                if ' ' in gene:
                    gene = gene.split()[0]
            else:
                pattern = config_mt["feature_sequences"]["replace"]["in"]
                repl = config_mt["feature_sequences"]["replace"]["out"]
                gene = re.sub(pattern, repl, gene)

            # NOTE: uncomment to debug feature sequences
            #import ipdb; ipdb.set_trace()

            if gene == '':
                continue

            if gene in features:
                seqs[featuress.at[gene]] = seq


# CREDIT NOTE: FROM BIOPYTHON
def _SimpleFastaParser(handle):
    """Iterate over Fasta records as string tuples.

    Arguments:
     - handle - input stream opened in text mode

    For each record a tuple of two strings is returned, the FASTA title
    line (without the leading '>' character), and the sequence (with any
    whitespace removed). The title line is not divided up into an
    identifier (the first word) and comment or description.

    >>> with open("Fasta/dups.fasta") as handle:
    ...     for values in SimpleFastaParser(handle):
    ...         print(values)
    ...
    ('alpha', 'ACGTA')
    ('beta', 'CGTC')
    ('gamma', 'CCGCC')
    ('alpha (again - this is a duplicate entry to test the indexing code)', 'ACGTA')
    ('delta', 'CGCGC')

    """
    # Skip any text before the first record (e.g. blank lines, comments)
    for line in handle:
        if line[0] == ">":
            title = line[1:].rstrip()
            break
    else:
        # no break encountered - probably an empty file
        return

    # Main logic
    # Note, remove trailing whitespace, and any internal spaces
    # (and any embedded \r which are possible in mangled files
    # when not opened in universal read lines mode)
    lines = []
    for line in handle:
        if line[0] == ">":
            yield title, "".join(lines).replace(" ", "").replace("\r", "")
            lines = []
            title = line[1:].rstrip()
            continue
        lines.append(line.rstrip())

    yield title, "".join(lines).replace(" ", "").replace("\r", "")

