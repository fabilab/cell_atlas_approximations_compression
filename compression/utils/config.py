import pathlib
import yaml

from .paths import (
    root_repo_folder,
    raw_atlas_folder,
)


def load_config(species):
    config_path = pathlib.Path(__file__).parent.parent / "organism_configs" / (species + ".yml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Propagate cell supertypes and order for simplicity
    if 'cell_annotations' in config:
        for mt in config['measurement_types']:
            config_mt = config[mt]
            if 'cell_annotations' not in config_mt:
                config_mt['cell_annotations'] = config['cell_annotations']
            else:
                for st in config['cell_annotations']:
                    config_mt['cell_annotations'][st] = config['cell_annotations'][st]

    for mt in config["measurement_types"]:
        config_mt = config[mt]

        if ("path" not in config_mt) and ("path_global" not in config_mt):
            config_mt["path"] = species + '.h5ad'

        if ("path" in config_mt) and isinstance(config_mt["path"], str):
            config_mt["path"] = {t: config_mt["path"] for t in config_mt["tissues"]}

        # Use absolute paths
        root_fdn = raw_atlas_folder / mt / species
        for key in ["path_global", "path_metadata_global"]:
            if key in config_mt:
                config_mt[key] = root_fdn / config_mt[key]
        if "path" in config_mt:
            for tissue in config_mt["path"]:
                config_mt["path"][tissue] = root_fdn / config_mt["path"][tissue]

        if "filter_cells" not in config_mt:
            config_mt["filter_cells"] = {}

        # NOTE: These could be defaulted to None, but this way we at least get an idea about what
        # kinds of data structures we are talking about. Not a good substitute for proper
        # documentation, I know.
        if "require_subannotation" not in config_mt["cell_annotations"]:
            config_mt["cell_annotations"]["require_subannotation"] = []

        if "subannotation_kwargs" not in config_mt["cell_annotations"]:
            config_mt["cell_annotations"]["subannotation_kwargs"] = {}

        if "blacklist" not in config_mt["cell_annotations"]:
            config_mt["cell_annotations"]["blacklist"] = {}

        if "tissue_restricted" not in config_mt["cell_annotations"]:
            config_mt["cell_annotations"]["tissue_restricted"] = {}
        # END OF NOTE

        celltype_order = []
        for supertype in config_mt["cell_annotations"]["supertype_order"]:
            ct_order_supertype = (
                supertype, config_mt["cell_annotations"]["cell_supertypes"][supertype],
            )
            celltype_order.append(ct_order_supertype)
        config_mt["cell_annotations"]["celltype_order"] = celltype_order

        del config_mt["cell_annotations"]["supertype_order"]
        del config_mt["cell_annotations"]["cell_supertypes"]

        if "feature_annotation" not in config_mt:
            config_mt["feature_annotation"] = False
        else:
            # FIXME
            config_mt["feature_annotation"] = root_repo_folder / 'data' / 'gene_annotations' / config_mt["feature_annotation"]

    return config


