"""Analyse the output of SATURN

This script requires anndata and scanpy. One way to do that is to use the SATURN conda environment:

source ~/miniconda3/bin/activate && conda activate saturn

"""

import os
import sys
import pathlib
from collections import defaultdict
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from Bio import Phylo
import torch


plants = [
    "t_aestivum",
    "f_vesca",
    "a_thaliana",
    "l_minuta",
    "z_mays",
]


full_name_dict = {
    "a_queenslandica": "Amphimedon queenslandica",
    "h_sapiens": "Homo sapiens",
    "m_musculus": "Mus musculus",
    "m_murinus": "Microcebus murinus",
    "d_rerio": "Danio rerio",
    "x_laevis": "Xenopus laevis",
    "t_adhaerens": "Trichoplax adhaerens",
    "s_lacustris": "Spongilla lacustris",
    "d_melanogaster": "Drosophila melanogaster",
    "l_minuta": "Lemna minuts",
    "a_thaliana": "Arabidopsis thaliana",
    "z_mays": "Zea mays",
    "f_vesca": "Fragaria vesca",
    "o_sativa": "Oryza sativa",
    "c_elegans": "Caenorhabditis elegans",
    "s_purpuratus": "Strongylocentrotus purpuratus",
    "s_pistillata": "Stylophora pistillata",
    "i_pulchra": "Isodiametra pulchra",
    "c_gigas": "Crassostrea gigas",
    "c_hemisphaerica": "Clytia hemisphaerica",
    "h_miamia": "Hofstenia miamia",
    "m_leidyi": "Mnemiopsis leidyi",
    "n_vectensis": "Nematostella vectensis",
    "p_crozieri": "Pseudoceros crozieri",
    "p_dumerilii": "Platynereis dumerilii",
    "s_mansoni": "Schistosoma mansoni",
    "s_mediterranea": "Schmidtea mediterranea",
    "t_aestivum": "Triticum aestivum",
}
full_name_dict_rev = {val: key for key, val in full_name_dict.items()}


def find_otts(names):
    import requests

    full_names = [full_name_dict.get(name, name) for name in names]
    resp = requests.post(
        "https://api.opentreeoflife.org/v3/tnrs/match_names",
        json=dict(names=full_names),
    ).json()
    ott_ids = {}
    for item in resp["results"]:
        name = item["name"]
        if name in ott_ids:
            continue
        matches = item["matches"]
        if len(matches) == 0:
            continue
        ott_id = matches[0]["taxon"]["ott_id"]
        ott_ids[name] = ott_id
    return ott_ids


def add_node_height_depth(tree):
    # Depth
    tree.root.depth = 0
    for node in tree.get_nonterminals(order="preorder"):
        for child in node.clades:
            if child.branch_length is not None:
                child.depth = node.depth + child.branch_length
            else:
                child.depth = node.depth + 1
    # Height
    for i, leaf in enumerate(tree.get_terminals()):
        leaf.height = i
    for node in tree.get_nonterminals(order="postorder"):
        node.height = np.mean([child.height for child in node.clades])


def find_induced_subtree(ott_dict):
    import requests
    from io import StringIO
    from Bio import Phylo

    # NOTE: This is the closest node in the synthetic tree to s mediterranea
    ott_dict_new = dict(ott_dict)
    ott_dict_new["Schmidtea mediterranea"] = 223973

    names, otts = tuple(map(list, zip(*ott_dict_new.items())))
    resp = requests.post(
        "https://api.opentreeoflife.org/v3/tree_of_life/induced_subtree",
        json=dict(ott_ids=list(otts)),
    )
    newick = resp.json()["newick"]
    tree = Phylo.read(StringIO(newick), "newick")
    for leaf in tree.get_terminals():
        idx = leaf.name.find("_ott")
        leaf.ottid = int(leaf.name[idx + 4 :])
        leaf.name = leaf.name[:idx].replace("_", " ")
        # Autocorrect s_mediterranea
        if leaf.name == "Schmidtea":
            leaf.name = "Schmidtea mediterranea"
            leaf.ottid = ott_dict[leaf.name]
        # The tree is right that magallana is the new name, but papers use the old one so
        # Same for p_crozieri, with the addition of a typo in the crozieri/crozierae
        if leaf.name == "Magallana gigas":
            leaf.name = "Crassostrea gigas"
        if leaf.name == "Maritigrella crozieri":
            leaf.name = "Pseudoceros crozieri"

    # Swap mouse and monkeys for clarity
    for node in tree.get_nonterminals():
        # If the node hangs exactly those three species and is not a passthrough node, swap the children
        if len(node.clades) < 2:
            continue
        hanging_leaves = set([leaf.name for leaf in node.get_terminals()])
        if hanging_leaves == set(
            ["Mus musculus", "Microcebus murinus", "Homo sapiens"]
        ):
            node.clades = node.clades[::-1]
            break

    return tree


def prune_tree_passthrough(tree):
    """Prune bypass nodes."""

    def fun(node):
        if len(node.clades) == 0:
            return
        while len(node.clades) == 1:
            node.name = node.clades[0].name
            if hasattr(node.clades[0], "ottid"):
                node.ottid = node.clades[0].ottid
            node.clades = node.clades[0].clades
        for child in node.clades:
            if len(child.clades) > 0:
                fun(child)

    fun(tree.root)

    return None


def recalibrate_tree(tree, known_ca_times=None):
    """Recalibrate tree branch lengths with actual history."""
    if known_ca_times is None:
        known_ca_times = {
            ("h_sapiens", "m_murinus"): 60,
            ("h_sapiens", "m_musculus"): 90,
            ("h_sapiens", "x_laevis"): 360,
            ("h_sapiens", "d_rerio"): 450,
            ("h_sapiens", "d_melanogaster"): 490,
            ("d_melanogaster", "c_elegans"): 480,
            ("h_sapiens", "i_pulchra"): 550,
            ("h_sapiens", "a_thaliana"): 1200,
            ("i_pulchra", "s_mediterranea"): 550,
            ("h_sapiens", "t_adhaerens"): 650,
            ("h_sapiens", "a_queenslandica"): 660,
            ("s_lacustris", "a_queenslandica"): 500,
            ("h_sapiens", "m_leidyi"): 700,
            ("t_aestivum", "a_thaliana"): 150,
            ("t_aestivum", "z_mays"): 70,
            # The following nobody really knows, but we'll guess them
            # We don't use the numbers for much anyway, just context
            ("s_pistillata", "n_vectensis"): 200,
            ("s_mediterranea", "p_crozieri"): 200,
            ("s_mediterranea", "s_mansoni"): 300,
            ("s_mediterranea", "c_elegans"): 470,
            ("s_mediterranea", "p_dumerilii"): 460,
            ("c_gigas", "p_dumerilii"): 200,
            ("i_pulchra", "h_miamia"): 200,
            ("h_sapiens", "c_hemisphaerica"): 630,
            ("s_pistillata", "c_hemisphaerica"): 300,
        }

    leaf_dict = {leaf.name: leaf for leaf in tree.get_terminals()}

    for (sp1, sp2), mya in known_ca_times.items():
        if full_name_dict[sp1] not in leaf_dict:
            continue
        if full_name_dict[sp2] not in leaf_dict:
            continue
        node = tree.common_ancestor(
            leaf_dict[full_name_dict[sp1]], leaf_dict[full_name_dict[sp2]]
        )
        node.depth = 1200 - mya
    for leaf in tree.get_terminals():
        leaf.depth = 1200

    for node in tree.get_nonterminals():
        for child in node.clades:
            if not hasattr(child, "depth"):
                print(child.name, "has no depth")
                print(child.get_terminals())
            child.branch_length = child.depth - node.depth

    return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analyse the output of SATURN.")
    parser.add_argument(
        "--n-macro",
        type=int,
        help="Number of macrogenes",
        default=1000,
    )
    parser.add_argument(
        "--n-hvg",
        type=int,
        help="Number of highly variable  genes",
        default=2000,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs in metric learning",
        default=50,
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        help="Number of epochs in pretraining",
        default=100,
    )
    parser.add_argument(
        "--kind",
        choices=["pretrain", "final", "25", "50"],
        default="final",
        type=str,
        help="Whether to look at pretrained, intermediate, or final trained model output",
    )
    parser.add_argument(
        "--kingdom",
        choices=["animal", "plant", "all"],
        default="all",
        type=str,
        help="Whether to look at animals, plants, or all species",
    )
    parser.add_argument(
        "--umap-dim",
        choices=[2, 3],
        default=2,
        type=int,
        help="Whether to plot UMAP in 2D or 3D",
    )
    parser.add_argument(
        "--savefig",
        action="store_true",
        help="Whether to save the figures",
    )
    args = parser.parse_args()

    output_fdn = pathlib.Path(
        f"/mnt/data/projects/cell_atlas_approximations/reference_atlases/data/saturn/output_nmacro{args.n_macro}_nhvg{args.n_hvg}_epochs_p{args.pretrain_epochs}_m{args.epochs}"
    )
    if not output_fdn.exists():
        sys.exit(f"Output folder {output_fdn} does not exist")

    saturn_csv_fn = output_fdn / "in_csv.csv"
    df = pd.read_csv(saturn_csv_fn, index_col="species")

    if args.kind == "pretrain":
        saturn_h5ad = (
            output_fdn / "saturn_results" / "adata_pretrain.h5ad"
        )  # 25 epochs for now, fix this
    elif args.kind != "final":
        saturn_h5ad = (
            output_fdn / "saturn_results" / f"adata_ep_{args.kind}.h5ad"
        )  # 25 epochs for now, fix this
    else:
        saturn_h5ad = (
            output_fdn / "saturn_results" / "final_adata.h5ad"
        )  # 25 epochs for now, fix this
    genes_to_macrogenes_fn = output_fdn / "saturn_results" / "genes_to_macrogenes.pkl"

    print("Read h5ad")
    # Try reading the chached object with umap etc.
    saturn_h5ad_cached = saturn_h5ad.with_stem(
        f"{saturn_h5ad.stem}_cached_{args.kingdom}.h5ad"
    )
    if saturn_h5ad_cached.exists():
        adata = anndata.read_h5ad(saturn_h5ad_cached)
    else:
        adata = anndata.read_h5ad(saturn_h5ad)

        if args.kingdom == "animal":
            adata = adata[~adata.obs["species"].isin(plants)]
        elif args.kingdom == "plant":
            adata = adata[adata.obs["species"].isin(plants)]

        print("Add tissue information")
        adata.obs["organ"] = ""
        separate_h5ad_fdn = output_fdn.parent / "h5ads"
        for species, datum in df.iterrows():
            if species not in adata.obs["species"].cat.categories:
                continue
            print(species)
            h5ad_fn = pathlib.Path(datum["path"])
            if str(h5ad_fn).startswith("/srv/scratch"):
                h5ad_fn = pathlib.Path(
                    f"/mnt/data/projects/cell_atlas_approximations/reference_atlases/data/saturn/h5ads/{species}.h5ad"
                )
            if not h5ad_fn.exists():
                print(f"File {h5ad_fn} does not exist, skipping")
                continue
            adatas = anndata.read_h5ad(h5ad_fn)
            cell_ids_species = adata.obs_names[adata.obs["species"] == species]
            organ_species = adatas.obs.loc[cell_ids_species, "organ"]
            adata.obs.loc[cell_ids_species, "organ"] = organ_species
            del adatas
        adata.obs["organ"] = pd.Categorical(adata.obs["organ"])
        __import__("gc").collect()

        print("Now we can make obs unique")
        adata.obs_names_make_unique()

        print("PCA")
        sc.pp.pca(adata)

        print("KNN")
        sc.pp.neighbors(adata)

        print("UMAP")
        sc.tl.umap(adata, n_components=args.umap_dim)

        print("Leiden clustering")
        sc.tl.leiden(adata, n_iterations=2, flavor="igraph")

        print("Standardise some cell type names")

        def mapfun(ct):
            return {
                "filament": "filamentous",
                "glia": "glial",
                "parenchyma": "parenchymal",
            }.get(ct, ct)

        adata.obs["cell_type"] = pd.Categorical(
            adata.obs["ref_labels"].astype(str).map(mapfun)
        )

        print(f"Cache the h5ad for kingdom {args.kingdom} to file")
        adata.write(saturn_h5ad_cached)

    print("Build tree of life with the species, as a null context")
    species_all = adata.obs["species"].cat.categories
    species_full_names = [full_name_dict[species] for species in species_all]
    ott_dict = find_otts(species_all)
    tree = find_induced_subtree(ott_dict)

    print("Prune passthrough nodes")
    prune_tree_passthrough(tree)

    print("Recalibrate tree with actual histotical times (roughly)")
    recalibrate_tree(tree)
    known_ca_times = {}
    for il, leaf1 in enumerate(tree.get_terminals()):
        for leaf2 in tree.get_terminals()[:il]:
            ca = tree.common_ancestor(leaf1, leaf2)
            known_ca_times[
                (full_name_dict_rev[leaf1.name], full_name_dict_rev[leaf2.name])
            ] = (1200 - ca.depth)

    if False:
        print("Find the remote cell types closest to groups of human cells")
        bait_groups = {
            ("h_sapiens", "endothelial"): ["arterial", "venous", "capillary", "CAP2"],
            ("h_sapiens", "muscle"): [
                "muscle",
                "smooth muscle",
                "striated muscle",
                "vascular smooth muscle",
            ],
            ("h_sapiens", "epithelial"): [
                "epithelial",
                "goblet",
                "brush",
                "crypt",
                "transit amp",
                "enterocyte",
                "paneth",
                "AT1",
                "AT2",
                "hillock-club",
                "hillock-basal",
                "club",
                "ciliated",
                "ductal",
                "acinar",
                "keratinocyte",
                "basal",
                "serous",
                "mucous",
                "cortical epi-like",
                "tuft",
                "melanocyte",
                "foveolar",
                "myoepithelial",
                "chief",
                "epidermal",
                "luminal",
                "parietal",
                "thyrocyte",
                "urothelial",
                "conjunctival",
                "corneal",
                "cholangiocyte",
                "mTEC",
            ],
            ("h_sapiens", "neuron"): ["neuron"],
            ("a_queenslandica", "choanocyte"): ["choanocyte"],
            ("a_queenslandica", "pinacocyte"): ["pinacocyte"],
        }
        bait_species, bait_group_label = "h_sapiens", "epithelial"
        bait_species, bait_group_label = "h_sapiens", "muscle"
        bait_species, bait_group_label = "a_queenslandica", "pinacocyte"
        print(f"Bait species and group: {bait_species}, {bait_group_label}")
        bait_cell_types = bait_groups[(bait_species, bait_group_label)]
        species_tgt = [
            "h_sapiens",
            "m_musculus",
            "m_murinus",
            "d_rerio",
            "x_laevis",
            "d_melanogaster",
            "a_queenslandica",
        ]
        k = 2
        adata_bait = adata[
            (adata.obs["species"] == bait_species)
            & (adata.obs["cell_type"].isin(bait_cell_types))
        ]
        Xbait = torch.tensor(adata_bait.X).to("cuda")
        adata_tgt = adata[adata.obs["species"].isin(species_tgt)]
        res = {}
        for species in species_tgt:
            if species == bait_species:
                continue
            adata_species = adata_tgt[adata_tgt.obs["species"] == species]
            if len(adata_species) == 0:
                continue
            print(species)
            Xspecies = torch.tensor(adata_species.X).to("cuda")
            cdis = torch.cdist(Xbait, Xspecies)
            mins = cdis.topk(k=k, axis=1, largest=False)
            anno_mins = adata_species.obs["cell_type"].iloc[
                mins.indices.to("cpu").numpy().ravel(),
            ]
            anno_mins_counts = anno_mins.value_counts()

            print("  Compute p-values against random bait cells")
            nrandom = 1000
            anno_mins_counts_randoms = []
            nbait = adata_bait.shape[0]
            for ir in range(nrandom):
                # FIXME: use with replacement for smaller samples
                adata_bait_random = adata[(adata.obs["species"] == species)]
                idx = np.random.choice(
                    adata_bait_random.obs_names, nbait, replace=False
                )
                adata_bait_random = adata_bait_random[idx]
                Xbait_random = torch.tensor(adata_bait_random.X).to("cuda")
                cdis_random = torch.cdist(Xbait_random, Xspecies)
                mins_random = cdis_random.topk(k=k, axis=1, largest=False)
                anno_mins_counts_random = (
                    adata_species.obs["cell_type"]
                    .iloc[mins_random.indices.to("cpu").numpy().ravel(),]
                    .value_counts()
                )
                anno_mins_counts_random.name = ir
                anno_mins_counts_randoms.append(anno_mins_counts_random)
            anno_mins_counts_randoms = pd.concat(anno_mins_counts_randoms, axis=1)
            anno_mins_counts_randoms = anno_mins_counts_randoms.loc[
                anno_mins_counts.index
            ]

            res = (
                (anno_mins_counts_randoms.T > anno_mins_counts)
                .mean(axis=0)
                .to_frame(name="pval")
            )
            res["mean_count"] = anno_mins_counts_randoms.mean(axis=1)
            res["mean_ratio"] = (
                1.0
                / k
                * anno_mins_counts
                / adata_species.obs["cell_type"].value_counts()
            )
            res["pval_adj"] = (res["pval"] * len(res)).clip(0, 1)
            res = res.sort_values(["pval_adj", "mean_ratio"], ascending=[True, False])

            # Bonferroni correction
            print("  Cell type: Pvalue adjusted Mean ratio")
            for ct, row in res.iterrows():
                pval = row["pval_adj"]
                mr = row["mean_ratio"]
                if pval > 0.05:
                    break
                print(f"  {ct}: {pval:.2e} {mr:.1f}")

            # FIXME: ratios tend to prefer rarer cell types, make a proper null
            if False:
                anno_ratios = (
                    1.0
                    / k
                    * anno_mins_counts
                    / adata_species.obs["cell_type"].value_counts()
                )
                anno_ratios = anno_ratios.nlargest(4)
                for ct, ratio in anno_ratios.items():
                    print(f"  {ct}: {ratio:.2f}")

    print("Visualise")
    plt.ion()
    plt.close("all")

    if False:
        print("Check cnidocytes/venom in basal animals")
        venom_cell_types = ["cnidocyte", "venom"]
        venom_counts = (
            adata.obs.loc[adata.obs["cell_type"].isin(venom_cell_types)]
            .groupby(["species", "cell_type"], observed=True)
            .size()
        )
        species_venom = venom_counts.index.get_level_values("species").unique().values
        res = {}
        res_summary = {}
        frac_correct_topk = {}
        for i, (species, ct) in enumerate(venom_counts.index):
            adata_species_ct = adata[
                (adata.obs["species"] == species) & (adata.obs["cell_type"] == ct)
            ]
            X1 = torch.tensor(adata_species_ct.X).to("cuda")
            for species2 in species_venom:
                if species2 == species:
                    continue
                adata_species2 = adata[adata.obs["species"] == species2]
                X2 = torch.tensor(adata_species2.X).to("cuda")
                cdis = torch.cdist(X1, X2)
                closest_counts = adata_species2.obs.iloc[
                    cdis.argmin(axis=1).to("cpu").numpy()
                ]["cell_type"].value_counts()

                for k in range(1, 5):
                    topk_counts = (
                        cdis.topk(k, axis=1, largest=False).indices.to("cpu").numpy()
                    )
                    tmp = np.array(
                        [
                            pd.Index(x).isin(venom_cell_types).any()
                            for x in adata_species2.obs["cell_type"].values[topk_counts]
                        ]
                    ).mean()
                    frac_correct_topk[(species, ct, species2, k)] = tmp

                res[(species, ct, species2)] = closest_counts

                num_correct = closest_counts.loc[
                    closest_counts.index.isin(venom_cell_types)
                ].sum()
                num_all = closest_counts.sum()
                if (species, species2) not in res_summary:
                    res_summary[(species, species2)] = {
                        "correct": 0,
                        "total": 0,
                    }
                res_summary[(species, species2)]["correct"] += num_correct
                res_summary[(species, species2)]["total"] += num_all
        frac_correct_topk = pd.Series(frac_correct_topk).unstack()

        for (species, species2), datum in res_summary.items():
            correct = datum["correct"]
            total = datum["total"]
            datum["frac_correct"] = 1.0 * correct / total

    if False:
        print("Build a layered graph of cell type transitions in animals")
        adata_animal = adata[~adata.obs["species"].isin(plants)]
        species_animal = adata_animal.obs["species"].cat.categories
        tree_animal = find_induced_subtree(
            {full_name_dict[k]: ott_dict[full_name_dict[k]] for k in species_animal}
        )
        add_node_height_depth(tree_animal)

        print(" Measure time since CA with humans")
        species_bait = "h_sapiens"
        time_since_ca = {}
        leaf_dict = {
            full_name_dict_rev[leaf.name]: leaf for leaf in tree_animal.get_terminals()
        }
        for species in species_animal:
            node = tree_animal.common_ancestor(
                leaf_dict[species], leaf_dict[species_bait]
            )
            time_since_ca[species] = leaf_dict[species_bait].depth - node.depth
        time_since_ca = pd.Series(time_since_ca).sort_values(ascending=False)

        print("Reconstruct evolutionary cell type flow for an animal example")
        leaves_example = [
            "h_sapiens",
            "m_murinus",
            "m_musculus",
            "x_laevis",
            # "d_rerio",
            # "d_melanogaster",
            "h_miamia",
            "n_vectensis",
            # "s_lacustris",
            # "a_queenslandica",
            "m_leidyi",
        ]
        nodes = defaultdict(list)
        edge_distances = defaultdict(float)
        for i in range(len(leaves_example) - 1):
            species1, species2 = leaves_example[i : i + 2]
            adata1 = adata_animal[adata_animal.obs["species"] == species1]
            adata2 = adata_animal[adata_animal.obs["species"] == species2]
            nodes[species1] = list(adata1.obs["cell_type"].cat.categories)
            X1 = torch.tensor(adata1.X).to("cuda")
            X2 = torch.tensor(adata2.X).to("cuda")
            cdis = torch.cdist(X1, X2)
            cell_types1 = adata1.obs["cell_type"].cat.categories
            cell_types2 = adata2.obs["cell_type"].cat.categories
            for i1, ct1 in enumerate(cell_types1):
                idx1 = (adata1.obs["cell_type"] == ct1).values.nonzero()[0]
                for i2, ct2 in enumerate(cell_types2):
                    idx2 = (adata2.obs["cell_type"] == ct2).values.nonzero()[0]
                    # NOTE: Average distance between cell types (this is better than topk, because it's less sensitive to abundance differences)
                    edge_distances[(species1, ct1, species2, ct2)] = float(
                        cdis[idx1][:, idx2].mean().to("cpu")
                    )
        nodes[species2] = list(adata2.obs["cell_type"].cat.categories)
        edge_distances = pd.Series(edge_distances)
        tmp = edge_distances.reset_index(name="distance").rename(
            columns={
                "level_0": "species1",
                "level_1": "cell_type1",
                "level_2": "species2",
                "level_3": "cell_type2",
            }
        )
        gby = tmp.groupby(["species1", "cell_type1"])
        k = 2
        edge_weights = []
        for (species1, cell_type1), datum in gby:
            # Exponential with nn-discount like in UMAP
            dmin = datum["distance"].min()
            dist = datum["distance"] - dmin
            eta = dist.sort_values().iloc[k - 1]
            # In case k == 1
            eta = max(eta, 1e-5)
            weight = np.exp(-dist / eta)
            weight[weight < 0.2] = 0
            datum["weight"] = weight
            weight = datum.set_index(["species2", "cell_type2"])["weight"]
            for (species2, cell_type2), w in weight.items():
                edge_weights.append((species1, cell_type1, species2, cell_type2, w))
        edge_weights = pd.DataFrame(
            edge_weights,
            columns=["species1", "cell_type1", "species2", "cell_type2", "weight"],
        )
        edge_weights = edge_weights[edge_weights["weight"] > 0]

        # Try to optimise node order...
        def compute_incident_length(species, cell_type, normalise=False):
            j1 = nodes[species].index(cell_type)
            tmp1 = edge_weights.loc[
                (edge_weights["species1"] == species)
                & (edge_weights["cell_type1"] == cell_type)
            ]
            diffs = []
            for _, row in tmp1.iterrows():
                species2 = row["species2"]
                cell_type2 = row["cell_type2"]
                j2 = nodes[species2].index(cell_type2)
                diff = np.abs(j1 - j2) * row["weight"]
                diffs.append(diff)

            j2 = nodes[species].index(cell_type)
            tmp2 = edge_weights.loc[
                (edge_weights["species2"] == species)
                & (edge_weights["cell_type2"] == cell_type)
            ]
            for _, row in tmp2.iterrows():
                species1 = row["species1"]
                cell_type1 = row["cell_type1"]
                j1 = nodes[species1].index(cell_type1)
                diff = np.abs(j1 - j2) * row["weight"]
                diffs.append(diff)
            if not normalise:
                return sum(diffs)
            else:
                return np.mean(diffs)

        print("Greedy node reordering")
        niter = 2000
        nct = np.asarray([len(nodes[sp]) for sp in leaves_example])
        p = 1.0 * nct / nct.sum()
        for it in range(niter):
            species1 = np.random.choice(leaves_example, p=p)
            cell_type1, cell_type2 = np.random.choice(
                nodes[species1], size=2, replace=False
            )
            l1_before = compute_incident_length(species1, cell_type1)
            l2_before = compute_incident_length(species1, cell_type2)
            l_before = l1_before + l2_before
            # Try swapping
            j1 = nodes[species1].index(cell_type1)
            j2 = nodes[species1].index(cell_type2)
            nodes[species1][j1], nodes[species1][j2] = (
                nodes[species1][j2],
                nodes[species1][j1],
            )
            l1_after = compute_incident_length(species1, cell_type1)
            l2_after = compute_incident_length(species1, cell_type2)
            l_after = l1_after + l2_after

            # If it did not help, swap back
            if l_after > l_before:
                nodes[species1][j1], nodes[species1][j2] = (
                    nodes[species1][j2],
                    nodes[species1][j1],
                )

        print("Plot graph")

        def highlight(species, cell_type, wmin=0.9):
            def recur_fun(species, cell_type, direction):
                if direction in (0, 1):
                    tmp1 = edge_weights.loc[
                        (edge_weights["species1"] == species)
                        & (edge_weights["cell_type1"] == cell_type)
                        & (edge_weights["weight"] > wmin)
                        & (edge_weights["done"] == False)
                    ]
                    edge_weights.loc[tmp1.index, "highlight"] = 1
                    edge_weights.loc[tmp1.index, "done"] = True
                    for _, row in tmp1.iterrows():
                        species2 = row["species2"]
                        cell_type2 = row["cell_type2"]
                        recur_fun(species2, cell_type2, direction=1)
                if direction in (0, 2):
                    tmp2 = edge_weights.loc[
                        (edge_weights["species2"] == species)
                        & (edge_weights["cell_type2"] == cell_type)
                        & (edge_weights["weight"] > wmin)
                        & (edge_weights["done"] == False)
                    ]
                    edge_weights.loc[tmp2.index, "highlight"] = 1
                    edge_weights.loc[tmp2.index, "done"] = True
                    for _, row in tmp2.iterrows():
                        species1 = row["species1"]
                        cell_type1 = row["cell_type1"]
                        recur_fun(species1, cell_type1, direction=2)

            edge_weights["highlight"] = 0
            edge_weights["done"] = False
            recur_fun(species, cell_type, 0)
            del edge_weights["done"]

        fig, ax = plt.subplots(figsize=(20, 20))
        nct_max = max(len(nodes[sp]) for sp in leaves_example)
        gety = lambda y, nct: 1.0 * np.asarray(y) / (nct - 1) * (nct_max - 1)
        highlight("h_sapiens", "T", wmin=0.1)
        # Draw vertices
        for i1, species1 in enumerate(leaves_example):
            nct1 = len(nodes[species1])
            y1 = np.arange(len(nodes[species1]))
            x1 = i1 * np.ones(len(y1))
            ax.scatter(x1, gety(y1, nct1), color="black", s=15, zorder=5)
            for j1, cell_type1 in enumerate(nodes[species1]):
                ax.text(
                    i1,
                    gety(j1, nct1) - 0.1,
                    cell_type1,
                    ha="center",
                    va="top",
                    fontsize=6,
                    zorder=10,
                )
        # Draw edges
        for _, row in edge_weights.iterrows():
            if "highlight" in row.index:
                if row["highlight"] == 0:
                    continue
            species1 = row["species1"]
            species2 = row["species2"]
            cell_type1 = row["cell_type1"]
            cell_type2 = row["cell_type2"]
            nct1 = len(nodes[species1])
            nct2 = len(nodes[species2])
            weight = row["weight"]
            i1 = leaves_example.index(species1)
            i2 = leaves_example.index(species2)
            j1 = nodes[species1].index(cell_type1)
            j2 = nodes[species2].index(cell_type2)
            ax.plot(
                [i1, i2],
                [gety(j1, nct1), gety(j2, nct2)],
                lw=2 * weight,
                color="black",
                alpha=weight,
                zorder=4,
            )
        # Final settings
        ax.set_xticks(np.arange(len(leaves_example)))
        ax.set_xticklabels(leaves_example)
        ax.xaxis.set_inverted(True)
        fig.tight_layout()

    if False:
        print("Analyse leiden clusters across evolution")

        fig, ax = plt.subplots(figsize=(13, 10))
        sc.pl.umap(
            adata,
            color="leiden",
            add_outline=True,
            size=20,
            ax=ax,
        )
        fig.tight_layout()

        clu_name = "40"
        adata_clu = adata[adata.obs["leiden"] == clu_name]

        # NOTE: differential abundance just tells us that if species share a cell type name, they are also close... which is good but obvious
        df = (
            adata_clu.obs.groupby(["species", "cell_type"])
            .size()
            .unstack(0, fill_value=0)
        )
        df = df.loc[:, df.sum() > 10]
        adata_clu = adata_clu[adata_clu.obs["species"].isin(df.columns)]

        if False:
            g = sns.clustermap(
                np.log1p(df), standard_scale=1, xticklabels=True, yticklabels=True
            )
            g.figure.set_size_inches(10, 23)

        # Filter cells that are total outliers
        centroid = adata_clu.obsm["X_umap"].mean(axis=0)
        dis_centroid = ((adata_clu.obsm["X_umap"] - centroid) ** 2).sum(axis=1)
        dis_max = np.percentile(dis_centroid, 99)
        adata_clu = adata_clu[adata_clu.obs_names[dis_centroid <= dis_max]]

        # Macrogene DE is where it's at
        # 1. Try with pseudotime, starting from the cell that is farthest from the UMAP center
        i0 = (adata_clu.obsm["X_umap"] ** 2).sum(axis=1).argmax()
        adata_clu.uns["iroot"] = i0
        sc.tl.diffmap(adata_clu)
        sc.tl.dpt(adata_clu)

        fig, axs = plt.subplots(1, 4, figsize=(20, 4))
        sc.pl.umap(
            adata_clu,
            color="dpt_pseudotime",
            add_outline=True,
            size=20,
            ax=axs[0],
        )
        sc.pl.umap(
            adata_clu,
            color="cell_type",
            add_outline=True,
            size=20,
            ax=axs[1],
            groups=list(df.sum(axis=1).nlargest(7).index),
        )
        sc.pl.umap(
            adata_clu,
            color="species",
            add_outline=True,
            size=20,
            ax=axs[2],
        )
        ax = axs[3]
        palette = dict(
            zip(
                adata_clu.obs["species"].cat.categories, adata_clu.uns["species_colors"]
            )
        )
        dfpt = adata_clu.obs.groupby("species")["dpt_pseudotime"]
        for i, (species, datum) in enumerate(dfpt):
            x = np.array([0] + list(np.sort(datum.values)))
            y = 1 - np.linspace(0, 1, len(x))
            ax.step(x, y, label=species, color=palette[species], where="post")
        fig.suptitle(clu_name)
        fig.tight_layout()

    if False:
        print("Plot the tree")
        fig, ax = plt.subplots(figsize=(10, 15))
        Phylo.draw(
            tree,
            axes=ax,
            show_confidence=False,
            do_show=False,
            label_func=lambda x: (x.name if x.name in species_full_names else None),
        )
        fig.tight_layout()

    print("Track specific cell types across evolution")
    ct_groups = {
        "muscle": [
            "muscle",
            "smooth muscle",
            "striated muscle",
            "vascular smooth muscle",
            "muscle-like",
            "pinacocyte",
            "apendopinacocyte",
            "basopinacocyte",
            "incurrent pinacocyte",
            "body wall muscle",
            "gut muscle",
            "pharyngeal muscle",
            "larval muscle",
            "adductor muscle",
            "velum striated muscle",
            "ventral muscle",
            "subumbrellar striated muscle",
            "cardiomyocyte",
        ],
        "neuron": ["neuron", "neuroid", "neural crest"],
        "immune": [
            "monocyte",
            "macrophage",
            "dendritic",
            "basophil",
            "eosinophil",
            "mast",
            "T",
            "NK",
            "NKT",
            "B",
            "hemocyte",
            "bactericidal",  # Amphimedon, expresses homolog of PFR1/2
            "coelomocyte",  # c_elegans, expresses homolog of MAN2B1/2 (expressed in human by immune cells)
            "phagocyte",  # s_mediterranea, expresses homolog of CTSS/L etc, cathepsins which are peptide degradation enzymes used in lysosomes for innate immunity
            "cathepsin",  # p_crozieri, expresses homolog of FPR3 which activates neutrophils
            "immune",  # coral pistillata, expresses homologs of IRF1 and MVP which are both innate immunity; also sea urchin, expresses a homolog of ARSA which is innate immunity
            "leukocyte",  # zebrafish, we should improve this annotation if possible
        ],
    }
    for gname, gcell_types in ct_groups.items():
        if gname != "muscle":
            continue
        tmp = (
            adata.obs.groupby(["species", "cell_type"]).size().unstack(0, fill_value=0)
        )
        species_group = tmp.loc[gcell_types].sum(axis=0) > 10
        species_group = species_group[species_group].index
        adata_group = adata[adata.obs["species"].isin(species_group)].copy()
        tree_group = find_induced_subtree(
            {full_name_dict[k]: ott_dict[full_name_dict[k]] for k in species_group}
        )
        prune_tree_passthrough(tree_group)
        recalibrate_tree(tree_group, known_ca_times=known_ca_times)
        add_node_height_depth(tree_group)

        # 3D plot, must do manually
        adata_group.obsm["X_umap"] = np.vstack(
            [adata_group.obsm["X_umap"].T, np.zeros(len(adata_group))]
        ).T
        for iz, leaf in enumerate(tree_group.get_terminals()):
            adata_group.obsm["X_umap"][
                adata_group.obs["species"] == full_name_dict_rev[leaf.name], 2
            ] = iz

        if True:
            print(" Plot UMAPs of this cell type group with a tree")

            fig = plt.figure(figsize=(17, 15))
            ax = fig.add_subplot(projection="3d")
            if gname == "neuron":
                palette = {
                    "neural crest": "purple",
                    "neuroid": "deeppink",
                    "neuron": "tomato",
                }
            else:
                palette = dict(
                    zip(
                        gcell_types,
                        sns.color_palette("husl", n_colors=len(gcell_types)),
                    )
                )
            for ct in adata_group.obs["cell_type"].cat.categories:
                if ct not in palette:
                    palette[ct] = (0.1, 0.1, 0.1, 0.003)
            legend_done = set()
            xmin, ymin = adata.obsm["X_umap"].min(axis=0)
            xmax, ymax = adata.obsm["X_umap"].max(axis=0)
            sc.pl.umap(
                adata_group,
                ax=ax,
                projection="3d",
                size=10,
                color="cell_type",
                palette=palette,
                groups=gcell_types,
                title=gname,
                na_color=(0.1, 0.1, 0.1, 0.003),
            )
            ax.set_zlabel("Species")
            ax.zaxis.set_label_position("lower")
            ax.set_zticks(np.arange(len(tree_group.get_terminals())))
            ax.set_zticklabels(
                [leaf.name for leaf in tree_group.get_terminals()],
                ha="left",
                style="italic",
            )
            ax.set_xlabel("UMAP 1", labelpad=5)
            ax.set_ylabel("UMAP 2", labelpad=5)
            ax.legend(
                loc="upper right",
                bbox_to_anchor=(0.20, 0.92),
                bbox_transform=ax.transAxes,
                ncol=1 + int(len(gcell_types) > 5),
                frameon=False,
                title="Cell type:",
            )

            # Plot tree
            dmax = max(leaf.depth for leaf in tree_group.get_terminals())
            xbase = adata.obsm["X_umap"][:, 0].min()
            xscale = 0.3 * (adata.obsm["X_umap"][:, 0].max() - xbase) / dmax

            def fun_xnew(x):
                return xbase + (x - dmax) * xscale

            for node in tree_group.get_nonterminals(order="preorder"):
                x0 = fun_xnew(node.depth)
                z0 = node.height
                for child in node.clades:
                    x1 = fun_xnew(child.depth)
                    z1 = child.height
                    ax.plot([x0, x0, x1], [0] * 3, zs=[z0, z1, z1], color="k", lw=2)
            # Tree scale
            x0, x1 = list(map(fun_xnew, [0, dmax]))
            xremote = fun_xnew(-0.2 * dmax)
            zheight = -1
            ax.plot([x0, x1], [0] * 2, zs=[zheight] * 2, color="k", lw=1.25)
            ax.plot(
                [xremote, x0], [0] * 2, zs=[zheight] * 2, color="k", lw=1.25, ls="--"
            )
            dints = [0, 100, 200, 300, 400, 500, 600, 700]
            for dint in dints:
                x = fun_xnew(dmax - dint)
                ax.plot(
                    [x] * 2,
                    [0] * 2,
                    zs=[
                        zheight - 0.25 + 0.125 * ((dint // 100) % 2),
                        zheight + 0.25 - 0.125 * ((dint // 100) % 2),
                    ],
                    color="k",
                    lw=1.25 - 0.25 * ((dint // 100) % 2),
                )
                if (dint // 100) % 2 == 0:
                    ax.text(x, 0, zheight - 1, f"{dint}", ha="center", va="top")
            ax.text(x0 + 0.1 * (x1 - x0), 0, zheight - 3, "Time [Mya]", "x", va="top")

            fig.tight_layout()
            if args.savefig:
                fig.savefig(
                    f"../figures/umap_3d_tree_{gname}.svg",
                )
                fig.savefig(
                    f"../figures/umap_3d_tree_{gname}.png",
                    dpi=300,
                )

        if True:
            print(" Track pseudotime along evolutionary time in this cell type group")
            species_bait = "h_sapiens"
            time_since_ca = {}
            leaf_dict = {
                full_name_dict_rev[leaf.name]: leaf
                for leaf in tree_group.get_terminals()
            }
            for species in species_group:
                node = tree_group.common_ancestor(
                    leaf_dict[species], leaf_dict[species_bait]
                )
                time_since_ca[species] = leaf_dict[species_bait].depth - node.depth
            time_since_ca = pd.Series(time_since_ca)
            oldest_species = time_since_ca.idxmax()
            adata_group_strict = adata_group[
                adata_group.obs["cell_type"].isin(gcell_types)
            ].copy()
            adata_group_other = adata_group[
                ~adata_group.obs["cell_type"].isin(gcell_types)
            ]
            umap_centre = adata_group_strict.obsm["X_umap"].mean(axis=0)
            adata_group_oldest = adata_group_strict[
                adata_group_strict.obs["species"] == oldest_species
            ]
            cell_zero = adata_group_oldest.obs_names[
                ((adata_group_oldest.obsm["X_umap"] - umap_centre) ** 2)
                .sum(axis=1)
                .argmin()
            ]
            idx_cell_zero = list(adata_group_strict.obs_names).index(cell_zero)
            adata_group_strict.uns["iroot"] = idx_cell_zero
            # NOTE: we need to recompute neighbors, otherwise the graph could be disconnected (by other cell types)
            # and even the connected part will have crazy weights (re UMAP onto cells that are not in the strict group)
            del (
                adata_group_strict.obsp["connectivities"],
                adata_group_strict.obsp["distances"],
            )
            sc.pp.neighbors(adata_group_strict)
            sc.tl.diffmap(adata_group_strict)
            sc.tl.dpt(adata_group_strict)
            dpt_mins = adata_group_strict.obs.groupby("species")["dpt_pseudotime"].min()
            dpt_maxs = adata_group_strict.obs.groupby("species")["dpt_pseudotime"].max()
            adata_group_strict.obs["dpt_pseudotime_stdscale"] = (
                adata_group_strict.obs["dpt_pseudotime"]
                - adata_group_strict.obs["species"].map(dpt_mins).astype(float)
            ) / (
                adata_group_strict.obs["species"].map(dpt_maxs).astype(float)
                - adata_group_strict.obs["species"].map(dpt_mins).astype(float)
            )

            print(" Plot pseudotime along the tree")
            fig = plt.figure(figsize=(17, 15))
            ax = fig.add_subplot(projection="3d")
            xmin, ymin = adata.obsm["X_umap"].min(axis=0)
            xmax, ymax = adata.obsm["X_umap"].max(axis=0)
            sc.pl.umap(
                adata_group_other,
                ax=ax,
                projection="3d",
                size=10,
                groups=[],
                na_color=(0.1, 0.1, 0.1, 0.003),
            )
            sc.pl.umap(
                adata_group_strict,
                ax=ax,
                projection="3d",
                size=10,
                color="dpt_pseudotime_stdscale",
                vmin=0,
                vmax=1,
                cmap="viridis",
                title=f"{gname}, phylogeny-adjusted pseudotime",
                # colorbar_loc=None,
            )
            x0, y0, z0 = adata_group_strict.obsm["X_umap"][idx_cell_zero]
            ax.scatter([x0], [y0], [z0], s=130, color="black", marker="*", zorder=11)
            ax.set_zlabel("Species")
            ax.zaxis.set_label_position("lower")
            ax.set_zticks(np.arange(len(tree_group.get_terminals())))
            ax.set_zticklabels(
                [leaf.name for leaf in tree_group.get_terminals()], ha="left"
            )
            # Plot tree
            dmax = max(leaf.depth for leaf in tree_group.get_terminals())
            xbase = adata.obsm["X_umap"][:, 0].min()
            xscale = 0.3 * (adata.obsm["X_umap"][:, 0].max() - xbase) / dmax
            for node in tree_group.get_nonterminals(order="preorder"):
                x0 = xbase + (node.depth - dmax) * xscale
                z0 = node.height
                y = 0
                for child in node.clades:
                    x1 = xbase + (child.depth - dmax) * xscale
                    z1 = child.height
                    ax.plot([x0, x0, x1], [y] * 3, zs=[z0, z1, z1], color="k", lw=2)
            ax.set_xlabel("UMAP 1", labelpad=5)
            ax.set_ylabel("UMAP 2", labelpad=5)

            # Plot tree
            dmax = max(leaf.depth for leaf in tree_group.get_terminals())
            xbase = adata.obsm["X_umap"][:, 0].min()
            xscale = 0.3 * (adata.obsm["X_umap"][:, 0].max() - xbase) / dmax

            def fun_xnew(x):
                return xbase + (x - dmax) * xscale

            for node in tree_group.get_nonterminals(order="preorder"):
                x0 = fun_xnew(node.depth)
                z0 = node.height
                for child in node.clades:
                    x1 = fun_xnew(child.depth)
                    z1 = child.height
                    ax.plot([x0, x0, x1], [0] * 3, zs=[z0, z1, z1], color="k", lw=2)
            # Tree scale
            x0, x1 = list(map(fun_xnew, [0, dmax]))
            xremote = fun_xnew(-0.2 * dmax)
            zheight = -1
            ax.plot([x0, x1], [0] * 2, zs=[zheight] * 2, color="k", lw=1.25)
            ax.plot(
                [xremote, x0], [0] * 2, zs=[zheight] * 2, color="k", lw=1.25, ls="--"
            )
            dints = [0, 100, 200, 300, 400, 500, 600, 700]
            for dint in dints:
                x = fun_xnew(dmax - dint)
                ax.plot(
                    [x] * 2,
                    [0] * 2,
                    zs=[
                        zheight - 0.25 + 0.125 * ((dint // 100) % 2),
                        zheight + 0.25 - 0.125 * ((dint // 100) % 2),
                    ],
                    color="k",
                    lw=1.25 - 0.25 * ((dint // 100) % 2),
                )
                if (dint // 100) % 2 == 0:
                    ax.text(x, 0, zheight - 1, f"{dint}", ha="center", va="top")
            ax.text(x0 + 0.1 * (x1 - x0), 0, zheight - 3, "Time [Mya]", "x", va="top")

            cax = fig.get_axes()[1]
            cax.set_position([0.82, 0.12, 0.1, 0.2])

            fig.tight_layout()
            if args.savefig:
                fig.savefig(
                    f"../figures/umap_3d_tree_{gname}_phylopseudotime.svg",
                )
                fig.savefig(
                    f"../figures/umap_3d_tree_{gname}phylopseudotime.png",
                    dpi=300,
                )

            if gname in ("muscle", "immune"):
                print("  Check distribution across human/mouse cell types")
                from scipy.stats import gaussian_kde

                palette = dict(
                    zip(
                        gcell_types,
                        sns.color_palette("husl", n_colors=len(gcell_types)),
                    )
                )
                cts_dict = {
                    "muscle": [
                        "smooth muscle",
                        "vascular smooth muscle",
                        "striated muscle",
                        "cardiomyocyte",
                    ],
                    "immune": [
                        "macrophage",
                        "dendritic",
                        "monocyte",
                        "T",
                        "NK",
                        "B",
                    ],
                }
                cts = cts_dict[gname][::-1]
                fig, axs = plt.subplots(
                    1, 3, figsize=(6, 1 + 0.3 * len(cts)), sharey=True
                )
                for ax, target_species in zip(
                    axs, ["h_sapiens", "m_murinus", "m_musculus"]
                ):
                    adata_tgt_strict = adata_group_strict[
                        adata_group_strict.obs["species"] == target_species
                    ]
                    gby = adata_tgt_strict.obs.groupby(
                        "cell_type"
                    ).dpt_pseudotime_stdscale
                    for ict, ct in enumerate(cts):
                        if ct not in gby.groups:
                            continue
                        datum = gby.get_group(ct)
                        color = palette[ct]
                        xinf = np.linspace(0, 1, 1000)
                        yinf = gaussian_kde(datum)(xinf)
                        yinf /= yinf.max() * 1.1
                        ax.fill_between(xinf, ict, ict + yinf, color=color, alpha=0.4)
                    ax.set_yticks(0.45 + np.arange(ict + 1))
                    if ax == axs[0]:
                        ax.set_yticklabels(cts)
                    ax.set_title(target_species)
                fig.text(0.52, 0.07, "Phylogeny-adjusted pseudotime", ha="center")
                fig.suptitle(f"{gname} cells")
                fig.tight_layout(w_pad=0.1, rect=[0, 0.11, 1, 1])
                if args.savefig:
                    fig.savefig(
                        f"../figures/kde_{gname}_phylopseudotime.svg",
                    )
                    fig.savefig(
                        f"../figures/kde_{gname}phylopseudotime.png",
                        dpi=300,
                    )

                print("3D version of the same plot")
                species_plot = ["h_sapiens", "m_murinus", "m_musculus"]
                fig = plt.figure(figsize=(7, 5))
                ax = fig.add_subplot(projection="3d")
                for isp, target_species in enumerate(species_plot):
                    adata_tgt_strict = adata_group_strict[
                        adata_group_strict.obs["species"] == target_species
                    ]
                    gby = adata_tgt_strict.obs.groupby(
                        "cell_type"
                    ).dpt_pseudotime_stdscale
                    for ict, ct in enumerate(cts):
                        if ct not in gby.groups:
                            continue
                        datum = gby.get_group(ct)
                        color = palette[ct]
                        xinf = np.linspace(0, 1, 1000)
                        yinf = gaussian_kde(datum)(xinf)
                        yinf /= yinf.max() * 1.1

                        idx = yinf > 0.01
                        xinf = xinf[idx]
                        yinf = yinf[idx]

                        ax.fill_between(
                            xinf,
                            isp,
                            ict,
                            xinf,
                            isp,
                            ict + yinf,
                            color=color,
                            alpha=0.8,
                        )
                ax.set_xlabel("Phylogeny-adjusted pseudotime")
                ax.set_yticks(np.arange(len(species_plot)))
                ax.set_yticklabels(species_plot, ha="left")
                ax.set_zticks(np.arange(ict + 1))
                ax.set_zticklabels(cts, ha="right", va="bottom")
                ax.zaxis.set_ticks_position("lower")
                ax.set_title(f"{gname} cells")
                ax.view_init(elev=10.0, azim=-75, roll=0)
                fig.tight_layout()
                if args.savefig:
                    fig.savefig(
                        f"../figures/kde_{gname}_phylopseudotime_3D.svg",
                    )
                    fig.savefig(
                        f"../figures/kde_{gname}phylopseudotime_3D.png",
                        dpi=300,
                    )

    if True:
        print("Start looking for macrogene changes along the tree")
        species_bait, cell_types_bait = "h_sapiens", [
            # "smooth muscle",
            # "vascular smooth muscle",
            "striated muscle",
            "cardiomyocyte",
        ]
        print(
            f" Find macrogene markers for a group of cell types in bait organism {species_bait}"
        )
        adata_bait = adata[adata.obs["species"] == species_bait]
        adata_bait_focal = adata_bait[adata_bait.obs["cell_type"].isin(cell_types_bait)]
        adata_bait_rest = adata_bait[~adata_bait.obs["cell_type"].isin(cell_types_bait)]
        Xmacro_focal = adata_bait_focal.obsm["macrogenes"]
        Xmacro_rest = adata_bait_rest.obsm["macrogenes"]
        frac_focal = (Xmacro_focal > 0).mean(axis=0)
        frac_rest = (Xmacro_rest > 0).mean(axis=0)
        delta_frac = pd.Series(
            frac_focal - frac_rest, index=np.arange(Xmacro_focal.shape[1])
        )
        # Macrogene markers for the cell type group
        markers_mg = delta_frac.nlargest(50).index

        print(" Verify that these macrogenes are influenced by genes that make sense")
        genes_to_macrogenes = pd.read_pickle(genes_to_macrogenes_fn)
        var_names_bait = np.array(
            [
                x[len(species_bait) + 1 :]
                for x in genes_to_macrogenes
                if x.startswith(species_bait)
            ]
        )
        genes_to_macrogenes_bait_matrix = np.vstack(
            [genes_to_macrogenes[f"{species_bait}_{g}"] for g in var_names_bait],
        )
        genes_to_marker_mg = var_names_bait[
            genes_to_macrogenes_bait_matrix[:, markers_mg].argmax(axis=0)
        ]
        print(
            f"  {species_bait} genes influential for the marker macrogenes in this cell type group: ",
            ", ".join(sorted(set(genes_to_marker_mg))),
        )

        print(" Find how these macrogenes change in certain cell types along evolution")
        cell_types_group_expanded = [
            "muscle",
            "smooth muscle",
            "striated muscle",
            "vascular smooth muscle",
            "muscle-like",
            "pinacocyte",
            "apendopinacocyte",
            "basopinacocyte",
            "incurrent pinacocyte",
            "body wall muscle",
            "gut muscle",
            "pharyngeal muscle",
            "larval muscle",
            "adductor muscle",
            "velum striated muscle",
            "ventral muscle",
            "subumbrellar striated muscle",
            "cardiomyocyte",
        ]
        tmp = (
            adata.obs.groupby(["species", "cell_type"]).size().unstack(0, fill_value=0)
        )
        species_group = tmp.loc[cell_types_group_expanded].sum(axis=0) > 10
        species_group = species_group[species_group].index
        avgs, fracs = [], []
        for species in species_group:
            adata_tmp = adata[
                (adata.obs["species"] == species)
                & (adata.obs["cell_type"].isin(cell_types_group_expanded))
            ]
            avgs.append(adata_tmp.obsm["macrogenes"][:, markers_mg].mean(axis=0))
            fracs.append((adata_tmp.obsm["macrogenes"][:, markers_mg] > 0).mean(axis=0))
        avgs = pd.DataFrame(np.vstack(avgs), index=species_group, columns=markers_mg)
        fracs = pd.DataFrame(np.vstack(fracs), index=species_group, columns=markers_mg)

        add_node_height_depth(tree)
        time_since_ca = {}
        leaf_dict = {
            full_name_dict_rev[leaf.name]: leaf for leaf in tree.get_terminals()
        }
        for species in species_group:
            node = tree.common_ancestor(leaf_dict[species], leaf_dict[species_bait])
            time_since_ca[species] = leaf_dict[species_bait].depth - node.depth
        time_since_ca = pd.Series(time_since_ca).loc[fracs.index]

        print(" Try correlation btw time since CA and macrogene expression")
        from scipy.stats import spearmanr, pearsonr

        res = {}
        for img, mg in enumerate(markers_mg):
            r = pearsonr(time_since_ca, avgs.loc[:, mg])
            rho = spearmanr(time_since_ca, avgs.loc[:, mg])
            rfrac = pearsonr(time_since_ca, fracs.loc[:, mg])
            rhofrac = spearmanr(time_since_ca, fracs.loc[:, mg])
            res[(mg, "r")] = r[0]
            res[(mg, "pval_r")] = r[1]
            res[(mg, "rho")] = rho[0]
            res[(mg, "pval_rho")] = rho[1]
            res[(mg, "r_frac")] = rfrac[0]
            res[(mg, "pval_r_frac")] = rfrac[1]
            res[(mg, "rho_frac")] = rhofrac[0]
            res[(mg, "pval_rho_frac")] = rhofrac[1]
        res = pd.Series(res).unstack()

        fig, axs = plt.subplots(1, 2, figsize=(11, 3))
        nmg = 5
        macrogenes_plot = res.nsmallest(nmg, "pval_r").index
        macrogenes_plot = pd.Index([203, 52, 673])
        palette = sns.color_palette("Set2", n_colors=len(macrogenes_plot))
        x0s = {}
        for i, mg in enumerate(macrogenes_plot):
            x = 0.1 + time_since_ca
            y = avgs.loc[x.index, mg]
            y2 = fracs.loc[x.index, mg]
            tmp = pd.DataFrame({"x": x, "y": y, "y2": y2}).sort_values("x")
            x = tmp["x"]
            y = tmp["y"]
            y2 = tmp["y2"]
            genes_in_this_mg = (
                pd.Series(
                    genes_to_macrogenes_bait_matrix[:, mg],
                    index=var_names_bait,
                )
                .nlargest(5)
                .index
            )
            label = f"{mg} ({', '.join(genes_in_this_mg)})"
            axs[0].plot(x, y + 1e-4, "o-", lw=2, markersize=4, color=palette[i])
            axs[1].plot(
                x,
                y2 + 1e-4,
                "o-",
                lw=2,
                markersize=4,
                label=label,
                color=palette[i],
            )

            # Fit logistic
            from scipy.optimize import curve_fit

            def logifunc(x, A, x0, k):
                return A * (1 - 1 / (1 + np.exp(-k * (x - x0))))

            popt, pcov = curve_fit(logifunc, x, y2, p0=[0.3, 0, 0.1])
            x0s[mg] = popt[1]
            xfit = np.linspace(x.min(), x.max(), 500)
            yfit = logifunc(xfit, *popt)
            axs[1].plot(xfit, yfit, "--", color=palette[i])

        axs[0].grid(True)
        axs[1].grid(True)
        axs[0].set_xlabel("Time since CA")
        axs[1].set_xlabel("Time since CA")
        axs[0].set_ylabel("Average macrogene expression")
        axs[1].set_ylabel("Fraction of muscle cells\nexpressing macrogene")
        axs[1].legend(
            loc="upper left", bbox_to_anchor=(1, 1), bbox_transform=axs[1].transAxes
        )
        axs[0].set_yscale("log")
        axs[0].invert_xaxis()
        axs[1].invert_xaxis()
        fig.tight_layout()
        if args.savefig:
            fig.savefig(
                "../figures/muscle_macrogenes_time_since_CA.svg",
            )
            fig.savefig(
                "../figures/muscle_macrogenes_time_since_CA.png",
                dpi=300,
            )

        print("Same, but more compact")
        from scipy.optimize import curve_fit

        def ilogifunc(x, A, x0, k):
            return A * (1 - 1 / (1 + np.exp(-k * (x - x0))))

        fig, ax = plt.subplots(figsize=(4, 2.5))
        macrogenes_plot = pd.Index([52, 203, 673])
        palette = sns.color_palette("Set2", n_colors=len(macrogenes_plot))
        x0s = {}
        for i, mg in enumerate(macrogenes_plot):
            x = 0.1 + time_since_ca
            y = avgs.loc[x.index, mg]
            y2 = fracs.loc[x.index, mg]
            tmp = pd.DataFrame({"x": x, "y": y, "y2": y2}).sort_values("x")
            x = tmp["x"]
            y2 = tmp["y2"]
            genes_in_this_mg = (
                pd.Series(
                    genes_to_macrogenes_bait_matrix[:, mg],
                    index=var_names_bait,
                )
                .nlargest(5)
                .index
            )
            popt, pcov = curve_fit(
                ilogifunc,
                x,
                y2,
                p0=[0.3, 0, 0.1],
                bounds=([0, -np.inf, 0], [1, np.inf, np.inf]),
            )
            x0s[mg] = x0 = popt[1]
            label = f"MG {mg}"
            ax.plot(
                x,
                y2 + 1e-4,
                "o-",
                lw=2,
                markersize=4,
                label=label,
                color=palette[i],
            )

            # Fit logistic
            xfit = np.linspace(x.min(), x.max(), 500)
            yfit = ilogifunc(xfit, *popt)
            ax.plot(xfit, yfit, "--", color=palette[i])

        ax.grid(True)
        ax.set_xlabel("Time since CA")
        ax.set_ylabel("Fraction of cells\nexpressing macrogene")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
        ax.invert_xaxis()
        fig.tight_layout()
        if args.savefig:
            fig.savefig(
                "../figures/muscle_macrogenes_time_since_CA_compact.svg",
            )
            fig.savefig(
                "../figures/muscle_macrogenes_time_since_CA_compact.png",
                dpi=300,
            )

        fig, ax1 = plt.subplots(figsize=(1.2, 1.2))
        ax1.barh(np.arange(len(x0s)), [-x0s[x] for x in macrogenes_plot], color=palette)
        ax1.set_yticks(np.arange(len(x0s)))
        ax1.set_yticklabels(macrogenes_plot)
        ax1.set_ylabel("Macrogene")
        ax1.set_xlabel("$x_0$")
        ax1.invert_yaxis()
        fig.tight_layout()
        if args.savefig:
            fig.savefig(
                "../figures/muscle_macrogenes_time_since_CA_logistics.svg",
            )
            fig.savefig(
                "../figures/muscle_macrogenes_time_since_CA_logistics.png",
                dpi=300,
            )

        plot_data = defaultdict(list)
        plot_crosses = defaultdict(list)
        for leaf in tree.get_terminals():
            species = full_name_dict_rev[leaf.name]
            if species not in avgs.index:
                plot_crosses["x"].append(leaf.depth)
                plot_crosses["y"].append(leaf.height)
                continue
            plot_data["x"].append(leaf.depth)
            plot_data["y"].append(leaf.height)
            plot_data["exp"].append(avgs.loc[species])
            leaf.exp = avgs.loc[species]
        for node in tree.get_nonterminals(order="postorder"):
            if len(node.clades) < 2:
                if hasattr(node.clades[0], "exp"):
                    node.exp = node.clades[0].exp
                continue
            # NOTE: This is weighted sum from all leaves
            if False:
                dists = {}
                for leaf in tree.get_terminals():
                    if full_name_dict_rev[leaf.name] not in avgs.index:
                        continue
                    species = full_name_dict_rev[leaf.name]
                    ca = tree.common_ancestor(leaf, node)
                    dists[species] = (leaf.depth - ca.depth) + (node.depth - ca.depth)
                dists = pd.Series(dists).loc[avgs.index]
                weights = 1.0 / dists
                exp = (avgs.T * weights).T.sum(axis=0) / weights.sum()
            # NOTE: This is average of children
            if True:
                n = 0
                exp = 0
                for child in node.clades:
                    if hasattr(child, "exp"):
                        n += 1
                        exp += np.log10(child.exp + 1e-4)
                if n == 0:
                    continue
                exp /= n
                exp = 10**exp - 1e-4
                node.exp = exp

            plot_data["x"].append(node.depth)
            plot_data["y"].append(node.height)
            plot_data["exp"].append(exp)
        plot_data["x"] = np.array(plot_data["x"])
        plot_data["y"] = np.array(plot_data["y"])
        plot_data["exp"] = pd.DataFrame(
            np.array(plot_data["exp"]), columns=avgs.columns
        )
        plot_crosses["x"] = np.array(plot_crosses["x"])
        plot_crosses["y"] = np.array(plot_crosses["y"])

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        axs = axs.ravel()
        for ax, mg in zip(axs, res.nsmallest(4, "r").index):
            Phylo.draw(
                tree,
                axes=ax,
                show_confidence=False,
                do_show=False,
                label_func=lambda x: (
                    "  " + x.name if x.name in species_full_names else None
                ),
            )
            ax.scatter(
                plot_data["x"],
                plot_data["y"] + 1,
                s=50 * np.log10((plot_data["exp"][mg] + 1e-4) * 100) ** 2,
                color="k",
            )
            ax.scatter(
                plot_crosses["x"],
                plot_crosses["y"] + 1,
                marker="x",
                s=25,
                color="tomato",
            )
            genes_in_this_mg = (
                pd.Series(
                    genes_to_macrogenes_bait_matrix[:, mg],
                    index=var_names_bait,
                )
                .nlargest(5)
                .index
            )
            label = f"{mg} ({', '.join(genes_in_this_mg)})"
            ax.set_title(label)
        fig.tight_layout()

    if False:
        print(" Plot UMAPs of this cell type group without a tree")
        fig, axs = plt.subplots(3, 6, figsize=(18, 7), sharex=True, sharey=True)
        axs = axs.ravel()
        for species, ax in zip(species_group, axs):
            adata_species = adata[adata.obs["species"] == species]
            sc.pl.umap(
                adata_species,
                color="cell_type",
                add_outline=True,
                size=20,
                ax=ax,
                groups=gcell_types,
                legend_loc="best",
                frameon=False,
                title=species,
            )
        fig.suptitle(gname)
        fig.tight_layout()

    if args.umap_dim == 2:

        if False:
            sc.pl.umap(
                adata, color="species", title="Species", add_outline=True, size=20
            )
            fig = plt.gcf()
            fig.set_size_inches(9, 5)
            fig.tight_layout()
            # fig.savefig("../figures/combined_umap_saturn_atlasapprox_species.png", dpi=300)

            sc.pl.umap(adata, color="organ", title="Organ", add_outline=True, size=20)
            fig2 = plt.gcf()
            fig2.set_size_inches(10, 5)
            fig2.tight_layout()
            # fig2.savefig("../figures/combined_umap_saturn_atlasapprox_organ.png", dpi=300)

            cell_types = np.sort(adata.obs["cell_type"].unique())
            colors = sns.color_palette("husl", n_colors=len(cell_types))
            palette = dict(zip(cell_types, colors))
            sc.pl.umap(
                adata,
                color="cell_type",
                title="Cell Type",
                add_outline=True,
                size=15,
                palette=dict(zip(cell_types, colors)),
            )
            fig3 = plt.gcf()
            fig3.set_size_inches(17, 9.8)
            fig3.axes[0].legend(
                ncol=5,
                fontsize=6,
                bbox_to_anchor=(1, 1),
                bbox_transform=fig3.axes[0].transAxes,
            )
            fig3.tight_layout()
            # fig3.savefig("../figures/combined_umap_saturn_atlasapprox_celltype.png", dpi=300)

        cell_types = np.sort(adata.obs["cell_type"].unique())
        colors = sns.color_palette("husl", n_colors=len(cell_types))
        palette = dict(zip(cell_types, colors))
        for i, ct in enumerate(cell_types):
            print(i + 1, ct)
            sc.pl.umap(
                adata,
                color="cell_type",
                title="Cell Type",
                groups=[ct],
                add_outline=True,
                size=18,
                palette=palette,
            )
            fig4 = plt.gcf()
            fig4.set_size_inches(8, 5)
            fig4.axes[0].legend(
                ncol=1,
                fontsize=6,
                bbox_to_anchor=(1, 1),
                bbox_transform=fig4.axes[0].transAxes,
            )
            fig4.savefig(f"../figures/single_cell_types/{ct}.png", dpi=300)
            plt.close(fig4)

        # fig3, axs = plt.subplots(3, 4, figsize=(12, 9))
        # axs = axs.ravel()
        # palette = {
        #    1: 'tomato',
        #    0: (0.9, 0.9, 0.9, 0.001),
        # }
        # for species, ax in zip(species_full_dict.keys(), axs):
        #    adata.obs['is_focal'] = pd.Categorical((adata.obs['species'] == species).astype(int))
        #    sc.pl.umap(adata, color="is_focal", title=species_full_dict[species], add_outline=True, size=20, ax=ax, legend_loc=None, palette=palette, groups=[1], na_color=palette[0])
        # fig3.tight_layout()
        # fig3.savefig("../figures/combined_umap_saturn_all_species_first_try.png", dpi=600)

        # palette = {
        #    "soldier": "darkgrey",
        #    "worker": "purple",
        #    "king": "steelblue",
        #    "queen": "deeppink",
        #    "roach": "seagreen",
        # }
        # sc.pl.umap(adata, color="caste", title="Caste", add_outline=True, size=20, palette=palette)
        # fig4 = plt.gcf()
        # fig4.set_size_inches(6.5, 5)
        # fig4.tight_layout()
        # fig4.savefig("../figures/combined_umap_saturn_all_species_first_try_caste.png", dpi=600)

    else:
        raise NotImplementedError("3D visualisation not implemented yet")

    if args.kingdom != "plant":
        print("Check closest human cells")
        import torch

        obsm_space = "pca"
        adata_h = adata[adata.obs["species"] == "h_sapiens"]
        adata_nh = adata[adata.obs["species"] != "h_sapiens"]
        cell_types_h = np.sort(adata_h.obs["cell_type"].unique())
        cell_types_nh = np.sort(adata_nh.obs["cell_type"].unique())
        coords_h = adata_h.obsm[f"X_{obsm_space}"]
        coords_nh = adata_nh.obsm[f"X_{obsm_space}"]
        coords_nht = torch.tensor(coords_nh).to("cuda")
        coords_ht = torch.tensor(coords_h).to("cuda")

        k = 15
        knn = -np.ones((adata_nh.shape[0], k), dtype=np.int64)
        batch_size = 1000
        i = 0
        majority_closest = []
        while True:
            # Compute distances on the GPU
            norm = torch.cdist(coords_nht[i : i + batch_size], coords_ht)
            knni = norm.topk(k, largest=False).indices
            ct_knn = (
                adata_h.obs["cell_type"]
                .iloc[np.asarray(knni.ravel().to("cpu"))]
                .values.reshape(knni.shape)
            )
            ctu = np.unique(ct_knn.ravel())
            counts = np.zeros((len(knni), len(ctu)), np.int64)
            for j, ct in enumerate(ctu):
                counts[:, j] = (ct_knn == ct).sum(axis=1)
            maj = ctu[counts.argmax(axis=1)]
            majority_closest.append(maj)

            i += batch_size
            if i >= coords_nh.shape[0]:
                break

        majority_closest = np.concatenate(majority_closest)
        adata_nh.obs["closest_human"] = pd.Categorical(majority_closest)

        adata_nh.obs["closest_human"] = pd.Categorical(majority_closest)
        df = adata_nh.obs[["cell_type", "closest_human", "species"]]

        if False:
            from sklearn.metrics import ConfusionMatrixDisplay

            species = "m_musculus"
            skpl = ConfusionMatrixDisplay.from_predictions(
                *df.loc[
                    df["species"] == species, ["cell_type", "closest_human"]
                ].values.T,
                include_values=False,
                xticks_rotation="vertical",
                normalize="true",
            )
            fig5 = skpl.figure_
            fig5.set_size_inches(25, 25)
            fig5.tight_layout()

        from sklearn.metrics import confusion_matrix
        from scipy.cluster.hierarchy import linkage, leaves_list

        labels_unsorted = np.unique(df[["cell_type", "closest_human"]].values.ravel())
        conf_matrix = confusion_matrix(
            df["cell_type"], df["closest_human"], labels=labels_unsorted
        )
        conf_matrix_norm = 1.0 * (conf_matrix.T / (0.001 + conf_matrix.sum(axis=1))).T

        if False:
            link = linkage(
                1 - conf_matrix_norm, method="average", optimal_ordering=True
            )
            idx_labels_sorted = leaves_list(link)
            conf_matrix_sorted = conf_matrix_norm[idx_labels_sorted][
                :, idx_labels_sorted
            ]
            labels_sorted = labels_unsorted[idx_labels_sorted]
            fig6, ax = plt.subplots(figsize=(20, 20))
            ax.matshow(conf_matrix_sorted, cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(np.arange(len(labels_sorted)))
            ax.set_yticks(np.arange(len(labels_sorted)))
            # ax.set_xticklabels(labels_sorted, rotation=90)
            ax.set_xticklabels([])
            ax.set_yticklabels(labels_sorted, fontsize=6)
            fig6.tight_layout()

        labels_with_human = cell_types_nh[pd.Index(cell_types_nh).isin(cell_types_h)]
        idx_with_human = pd.Index(labels_unsorted).isin(labels_with_human).nonzero()[0]
        conf_matrix_with_human = conf_matrix[idx_with_human][:, idx_with_human]
        frac_exact = (
            1.0
            * conf_matrix_with_human[
                np.arange(len(conf_matrix_with_human)),
                np.arange(len(conf_matrix_with_human)),
            ].sum()
            / conf_matrix_with_human.sum()
        )
        tmp = conf_matrix_with_human.copy()
        tmp = 1.0 * (tmp.T / (tmp.sum(axis=1) + 0.001)).T
        tmp[
            np.arange(len(conf_matrix_with_human)),
            np.arange(len(conf_matrix_with_human)),
        ] = 0
        top_nondiag = np.unravel_index(np.argsort(tmp.ravel())[::-1][:10], tmp.shape)
        for i, j in zip(*top_nondiag):
            print(f"{labels_with_human[i]} -> {labels_with_human[j]}: {tmp[i, j]}")

        mat_to_plot = (
            1.0
            * conf_matrix_with_human.T
            / (conf_matrix_with_human.sum(axis=1) + 0.001)
        ).T
        link = linkage(1 - mat_to_plot, method="average", optimal_ordering=True)
        idx_with_human_sorted = leaves_list(link)
        labels_with_human_sorted = labels_with_human[idx_with_human_sorted]
        mat_to_plot_sorted = mat_to_plot[idx_with_human_sorted][
            :, idx_with_human_sorted
        ]
        fig7, ax = plt.subplots(figsize=(10, 10))
        ax.matshow(mat_to_plot_sorted, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(labels_with_human_sorted)))
        ax.set_yticks(np.arange(len(labels_with_human_sorted)))
        ax.set_xticklabels(labels_with_human_sorted, rotation=90, fontsize=6)
        ax.set_yticklabels(labels_with_human_sorted, fontsize=6)
        for xi in np.arange(len(labels_with_human_sorted) - 1):
            ax.axvline(xi + 0.5, color="w", linewidth=0.5)
            ax.axhline(xi + 0.5, color="w", linewidth=0.5)
        fig7.tight_layout()

        print("Set up cell type classifier")
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import RocCurveDisplay, auc
        from sklearn.preprocessing import LabelBinarizer
        import cupy as cp

        # TODO: exclude extremely rare cell types (<20 cells)
        cell_type_abundances = adata.obs["cell_type"].value_counts()
        cell_types_abu = cell_type_abundances[cell_type_abundances >= 20].index
        # It automatically eliminates empty categories
        adata_cv = adata[adata.obs["cell_type"].isin(cell_types_abu)].copy()

        X = adata_cv.X
        ycat = adata_cv.obs["cell_type"]
        y = ycat.cat.codes.values
        cell_types = ycat.cat.categories
        nct = len(cell_types)

        num_round = 10

        def build_model(modelname):
            if modelname == "xgb":
                return __import__("xgboost").XGBClassifier(
                    tree_method="hist", device="cuda", early_stopping_rounds=3
                )
            if modelname == "logistic":
                from sklearn.linear_model import LogisticRegression

                return LogisticRegression(max_iter=1000)

            raise NotImplementedError(f"Model {modelname} not implemented")

        print("Cross validation")
        # NOTE: logistic is much slower because on CPU. One could move it to the GPU with pytorch, check out:
        # https://diegoinacio.github.io/machine-learning-notebooks-page/pages/MCLR_PyTorch.html
        for model in ["logistic", "xgb"]:
            skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)
            fig, ax = plt.subplots(figsize=(6, 6))
            colors = sns.color_palette("husl", n_colors=5)
            for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
                print(f"Fold {fold}:")
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                clf = build_model(model)

                # xgb works on GPUs too
                if model == "xgb":
                    X_train_gpu = cp.array(X_train)
                    y_train_gpu = cp.array(y_train)
                    X_test_gpu = cp.array(X_test)
                    y_test_gpu = cp.array(y_test)
                    fit_kwargs = dict(eval_set=[(X_test_gpu, y_test_gpu)])

                else:
                    X_train_gpu = X_train
                    y_train_gpu = y_train
                    X_test_gpu = X_test
                    y_test_gpu = y_test
                    fit_kwargs = dict()

                # eval_set is only used for early stopping, which makes it much faster
                clf.fit(X_train_gpu, y_train_gpu, **fit_kwargs)

                y_score = clf.predict_proba(X_test_gpu)

                label_binarizer = LabelBinarizer().fit(y_train)
                y_onehot_test = label_binarizer.transform(y_test)

                viz = RocCurveDisplay.from_predictions(
                    y_onehot_test.ravel(),
                    y_score.ravel(),
                    name=f"micro-average OvR fold {fold}",
                    color=colors[fold],
                    alpha=0.5,
                    lw=1,
                    ax=ax,
                    plot_chance_level=(fold == 0),
                    # despine=True,
                )
                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            ax.plot(
                mean_fpr,
                mean_tpr,
                color="b",
                label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
                lw=2,
                alpha=0.8,
            )

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(
                mean_fpr,
                tprs_lower,
                tprs_upper,
                color="grey",
                alpha=0.2,
                label=r"$\pm$ 1 std. dev.",
            )

            ax.set(
                xlabel="False Positive Rate",
                ylabel="True Positive Rate",
                title="Mean micro-averaged One-vs-Rest\nReceiver Operating Characteristic",
            )
            ax.legend(loc="lower right")
            fig.tight_layout()
