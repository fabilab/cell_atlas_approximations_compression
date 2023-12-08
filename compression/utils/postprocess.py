import gc
import numpy as np
import pandas as pd


def homogenise_features(compressed_atlas):
    """Ensure all tissues use the same features"""
    if len(compressed_atlas['tissues']) == 1:
        for tissue, group in compressed_atlas['tissues'].items():
            compressed_atlas['features'] = group.pop('features')
        return

    features_all = sorted(
        set().union(*(set(g['features']) for g in compressed_atlas['tissues'].values())),
    )
    compressed_atlas['features'] = np.asarray(features_all)

    for tissue, group in compressed_atlas['tissues'].items():
        features = group.pop('features')
        if features.tolist() == features_all:
            continue

        subgroup = group['celltype']
        for i in range(2):
            for key in ['avg', 'frac']:
                if key not in subgroup:
                    continue

                X = subgroup[key]
                X_new = pd.DataFrame(
                    np.zeros(
                        (len(features_all), X.shape[1]),
                        dtype=X.values.dtype,
                    ),
                    index=features_all,
                    columns=X.columns,
                )
                X_new.loc[features] = X.values
                subgroup[key] = X_new

            if i == 0:
                subgroup = subgroup['neighborhood']
