"""Test inference with SATURN-like model on the termites/fly data.

This must be run inside the Python 3.10 saturn conda environment: source ~/miniconda3/bin/activate && conda activate saturn

"""

import os
import sys
import pickle
from collections import Counter, defaultdict
import pathlib
import pandas as pd
import subprocess as sp
import argparse
import numpy as np
import anndata
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import torch


species_list = [
    "a_queenslandica",
    "a_thaliana",
    "c_elegans",
    "c_gigas",
    "c_hemisphaerica",
    "d_melanogaster",
    "d_rerio",
    "h_miamia",
    "h_sapiens",
    "i_pulchra",
    "l_minuta",
    "m_leidyi",
    "m_murinus",
    "m_musculus",
    "n_vectensis",
    "p_crozieri",
    "p_dumerilii",
    "s_lacustris",
    "s_mansoni",
    "s_mediterranea",
    "s_pistillata",
    "t_adhaerens",
    "t_aestivum",
    "x_laevis",
    "z_mays",
]
leaveout_dict = {key: i + 1 for i, key in enumerate(species_list)}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train SATURN on the atlasapprox data."
    )
    parser.add_argument(
        "--n-macrogenes",
        type=int,
        default=6,
        help="Number of macrogenes (only used to find the model)",
    )
    parser.add_argument(
        "--n-hvg", type=int, default=13, help="Number of highly variable genes"
    )
    parser.add_argument(
        "--n-epochs", default=1, help="Number of epochs in metric learning"
    )
    parser.add_argument(
        "--n-pretrain-epochs", default=1, help="Number of epochs in pretraining"
    )
    parser.add_argument(
        "--species",
        type=str,
        help="Infer embedding for data of this species",
        default=None,
    )
    parser.add_argument(
        "--train", action="store_true", help="Whether to train the transfer model"
    )
    parser.add_argument(
        "--leaveout",
        type=str,
        required=True,
        help="Use leaveout-traned model without this species for inference.",
    )
    parser.add_argument(
        "--secondary-analysis", action="store_true", help="Perform secondary analysis"
    )
    parser.add_argument(
        "--random-weights",
        action="store_true",
        help="Initialise the top layer with random weights.",
    )
    parser.add_argument(
        "--encoder",
        choices=["pretrain", "metric"],
        default="metric",
        help="Which encoder to use",
    )
    parser.add_argument(
        "--guide-species", type=str, default=None, help="Guide species for inference"
    )
    args = parser.parse_args()

    if args.leaveout is not None and args.species is None:
        args.species = args.leaveout

    fasta_root_folder = pathlib.Path(
        "/mnt/data/projects/cell_atlas_approximations/reference_atlases/data/saturn/peptide_sequences/"
    )
    embedding_root_fdn = fasta_root_folder.parent / "esm_embeddings"
    embeddings_summary_fdn = embedding_root_fdn.parent / "esm_embeddings_summaries/"
    h5ad_fdn = embeddings_summary_fdn.parent / "h5ads"
    training_output_fdn = (
        embeddings_summary_fdn.parent
        / f"output_nmacro{args.n_macrogenes}_nhvg{args.n_hvg}_epochs_p{args.n_pretrain_epochs}_m{args.n_epochs}"
    )
    training_output_leaveout_fdn = (
        training_output_fdn.parent
        / f"{training_output_fdn.stem}_leaveout_{args.leaveout}"
    )

    # There is a "from_kdm" folder that contains the big models, just to make sure I don't delete them by mistake
    if not training_output_leaveout_fdn.exists():
        leaveout_n = leaveout_dict[args.leaveout]
        training_output_leaveout_fdn = (
            training_output_leaveout_fdn.parent
            / "from_kdm"
            / f"{training_output_fdn.stem}_leaveout_{leaveout_n}"
        )
        inference_fdn = training_output_leaveout_fdn / "inference_output"
    if not training_output_fdn.exists():
        # FIXME: use actual full model, for now we got this
        if (args.n_macrogenes == 1500) and (args.n_hvg == 4000):
            training_output_fdn = (
                embeddings_summary_fdn.parent
                / "from_kdm"
                / f"output_nmacro1300_nhvg3500_epochs_p{args.n_pretrain_epochs}_m{args.n_epochs}"
            )
        else:
            training_output_fdn = (
                embeddings_summary_fdn.parent
                / "from_kdm"
                / f"output_nmacro{args.n_macrogenes}_nhvg{args.n_hvg}_epochs_p{args.n_pretrain_epochs}_m{args.n_epochs}"
            )
        assert training_output_fdn.exists()

    print("Find h5ad after inference and training")
    trained_adata_path = (
        training_output_leaveout_fdn / "saturn_results" / "final_adata.h5ad"
    )
    full_adata_path = training_output_fdn / "saturn_results" / "final_adata.h5ad"
    for inf_adata_path in inference_fdn.iterdir():
        if args.species not in inf_adata_path.stem:
            continue
        if args.guide_species not in inf_adata_path.stem:
            continue
        if args.train and str(inf_adata_path).endswith("finetuned.h5ad"):
            break
        elif (not args.train) and str(inf_adata_path).endswith("zeroshot.h5ad"):
            break
    else:
        raise IOError("Inference output h5ad file not found")
    print(f"Inference adata path: {inf_adata_path}")
    print(f"Leaveout adata path: {trained_adata_path}")
    print(f"Full model adata path: {full_adata_path}")

    print("Load h5ad for inference, leaveout training, and full training")
    adata_inf = anndata.read_h5ad(inf_adata_path)
    adata_train = anndata.read_h5ad(trained_adata_path)
    adata_full = anndata.read_h5ad(full_adata_path)

    print("Limit training and full adata to the guide/closest species")
    adata_train = adata_train[
        adata_train.obs["species"] == adata_inf.uns["guide_species"]
    ]
    adata_full = adata_full[
        adata_full.obs["species"].isin([args.species, adata_inf.uns["guide_species"]])
    ]

    print("Concatenate inference and leaveout adata")
    adata = anndata.concat([adata_inf, adata_train])

    print("Now we can make obs unique")
    adata.obs_names_make_unique()

    adatad = {
        "full": adata_full,
        "xfer": adata,
    }

    print("Standardise some cell type names and set new column")

    def mapfun(ct):
        return {
            "filament": "filamentous",
            "glia": "glial",
            "parenchyma": "parenchymal",
        }.get(ct, ct)

    for adatai in adatad.values():
        adatai.obs["cell_type"] = pd.Categorical(
            adatai.obs["ref_labels"].astype(str).map(mapfun)
        )

    print("Compute scores of weak learning via triplets")
    self_hits_dict = {}
    self_hits_total_frac = {}
    for key, adatai in adatad.items():
        # Target species
        adata_tgt = adatai[adatai.obs["species"] != adata_inf.uns["guide_species"]]
        # Guide
        adata_gui = adatai[adatai.obs["species"] == adata_inf.uns["guide_species"]]

        # Compute mutual shortest distances
        X_tgt = torch.tensor(adata_tgt.X).to("cuda")
        X_gui = torch.tensor(adata_gui.X).to("cuda")
        cdis = torch.cdist(X_tgt, X_gui)
        closest_gui = cdis.min(axis=1).indices.cpu().numpy()
        closest_back = cdis.min(axis=0).indices.cpu().numpy()
        closest_roundtrip = closest_back[closest_gui]

        df = pd.DataFrame(index=adata_tgt.obs_names)
        df["cell_type"] = adata_tgt.obs["cell_type"].copy()
        df["closest_roundtrip"] = adata_tgt.obs["cell_type"].values[closest_roundtrip]
        df["c"] = 1

        count_roundtrip = (
            df.groupby(["cell_type", "closest_roundtrip"]).size().unstack(fill_value=0)
        )
        nct = count_roundtrip.sum(axis=1)
        frac_roundtrip = (1.0 * count_roundtrip.T / nct).T
        self_hits_dict[key] = pd.Series(
            frac_roundtrip.values.diagonal(),
            index=count_roundtrip.index,
        )
        self_hits_total_frac[key] = (
            1.0 * count_roundtrip.values.diagonal().sum() / count_roundtrip.values.sum()
        )
    self_hits = pd.DataFrame(self_hits_dict)
    self_hits.columns.name = "model"

    print("Add stats to summary file")
    summary_output_fn = (
        pathlib.Path(".").absolute().parent.parent
        / "data"
        / "saturn_xfer_results"
        / "fraction_self_hits.pkl"
    )
    os.makedirs(summary_output_fn.parent, exist_ok=True)
    current_pkl = {}
    if summary_output_fn.exists():
        with open(summary_output_fn, "rb") as f:
            current_pkl = pickle.load(f)
    current_pkl[(args.species, args.guide_species)] = {
        "self_hits": self_hits.to_dict(),
        "self_hits_total_frac": self_hits_total_frac,
    }
    with open(summary_output_fn, "wb") as f:
        pickle.dump(current_pkl, f)

    # Too much information
    if False:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        sns.barplot(
            data=self_hits.stack().reset_index(name="frac"),
            x="cell_type",
            y="frac",
            hue="model",
            ax=ax,
        )
        fig.tight_layout()
        plt.ion()
        plt.show()

    # This one is useful, pick a representative case
    if True:
        spec, spec2 = "h_sapiens", "m_musculus"
        self_hits = pd.DataFrame(current_pkl[(spec, spec2)]["self_hits"])
        self_hits_total_frac = current_pkl[(spec, spec2)]["self_hits_total_frac"]
        fig, ax = plt.subplots(figsize=(3, 3))
        palette = {"full": (0.1, 0.1, 0.1), "xfer": "tomato"}
        labels = {"full": "foundation\nmodel", "xfer": "leaveout +\nfine tune"}
        for col in self_hits.columns:
            ax.ecdf(
                self_hits[col],
                label=labels[col],
                complementary=True,
                color=palette[col],
            )
        ax.axvline(self_hits_total_frac["full"], color=palette["full"], linestyle="--")
        ax.axvline(self_hits_total_frac["xfer"], color=palette["xfer"], linestyle="--")
        ax.arrow(
            self_hits_total_frac["full"] + 0.02,
            0.5,
            -0.04 + self_hits_total_frac["xfer"] - self_hits_total_frac["full"],
            0,
            length_includes_head=True,
            head_width=0.02,
            head_length=0.03,
            overhang=0.1,
            color="black",
        )
        ax.set(
            xlabel="Fraction of correct\nroundtrips (by cell type)",
            ylabel=f"Cumulative over cell\ntypes in {args.species}",
        )
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            pathlib.Path(".").absolute().parent.parent
            / "figures"
            / "cumulative_over_celltype_frac_annotation_roundtrips_results.png",
            dpi=300,
        )

    # Unclear how good the model really is, only clear that is equal or better than the full model... maybe for supplementary
    if False:
        print("Compare several leaveouts and guides")
        cm_reds = LinearSegmentedColormap.from_list("Custom", [(0, 0, 0), (1, 0, 0)])
        cm_blues = LinearSegmentedColormap.from_list("Custom", [(0, 0, 0), (0, 0, 1)])
        cm = {"^": cm_reds, "v": cm_blues}
        plot_data = {"^": defaultdict(list), "v": defaultdict(list)}

        fig, ax = plt.subplots(1, 1, figsize=(3, 3.2))
        species = list(sorted(set([k[0] for k in current_pkl.keys()])))
        for i, spec in enumerate(species):
            for j, spec2 in enumerate(species):
                if spec == spec2:
                    continue
                frac_full = current_pkl[(spec, spec2)]["self_hits_total_frac"]["full"]
                frac_xfer = current_pkl[(spec, spec2)]["self_hits_total_frac"]["xfer"]
                diff = frac_xfer - frac_full
                size = 10 + 100 * np.abs(diff)
                key = "^" if diff > 0 else "v"
                color = cm[key](min(1.0, 4 * np.abs(diff)))
                plot_data[key]["x"].append(j)
                plot_data[key]["y"].append(i)
                plot_data[key]["s"].append(size)
                plot_data[key]["c"].append(color)
        for marker, plot_datum in plot_data.items():
            if len(plot_datum["x"]) == 0:
                continue
            x = np.asarray(plot_datum["x"])
            y = np.asarray(plot_datum["y"])
            c = np.asarray(plot_datum["c"])
            s = np.asarray(plot_datum["s"])
            ax.scatter(x, y, s=s**2, c=c, marker=marker)
        ax.set_xticks(np.arange(len(species)))
        ax.set_yticks(np.arange(len(species)))
        ax.set_xticklabels(species, rotation=90)
        ax.set_yticklabels(species)
        ax.set_xlabel("Guide species")
        ax.set_ylabel("Leaveout species")
        ax.set_xlim(-0.5, len(species) - 0.5)
        ax.set_ylim(len(species) - 0.5, -0.5)
        fig.tight_layout()

    if True:
        plot_data = defaultdict(list)
        fig, ax = plt.subplots(figsize=(1.8, 2.2))
        for (spec, spec2), datum in current_pkl.items():
            y1 = datum["self_hits_total_frac"]["full"]
            y2 = datum["self_hits_total_frac"]["xfer"]
            plot_data["x"].append(0)
            plot_data["y"].append(y1)
            plot_data["x"].append(1)
            plot_data["y"].append(y2)
            m = y2 - y1
            ax.arrow(
                0.1,
                y1 + m * 0.1,
                0.8,
                m * 0.8,
                head_width=0.05,
                head_length=0.05,
                color=(0, 0, 0, 0.5),
            )
        ax.scatter(
            plot_data["x"], plot_data["y"], s=30, alpha=0.8, c="black", zorder=10
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["foundation\nmodel", "leaveout +\nfine tune"])
        ax.grid(True)
        ax.set_ylabel("Fraction of correct\nannotation round trips")
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.5, 1.5)
        ax.set_yticks([0, 0.5, 1])
        fig.tight_layout()
        fig.savefig(
            pathlib.Path(".").absolute().parent.parent
            / "figures"
            / "frac_annotation_roundtrips_results.png",
            dpi=300,
        )

    plt.ion()
    plt.show()

    # FIXME:
    sys.exit()

    print("PCA")
    sc.pp.pca(adata)

    print("KNN")
    sc.pp.neighbors(adata)

    print("UMAP")
    sc.tl.umap(adata, n_components=2)

    print("Visualise")
    plt.ion()
    plt.close("all")

    fig3, axs = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    sc.pl.umap(
        adata,
        color="species",
        title="Species",
        add_outline=True,
        size=20,
        ax=axs[0],
    )
    axs[0].legend(ncol=1, fontsize=6, loc="best")
    for i, species in enumerate(
        [adata_inf.uns["guide_species"], f"{args.species}-inf"]
    ):
        ax = axs[i + 1]
        adata.obs["cell_type_tmp"] = adata.obs["cell_type"].astype(str)
        adata.obs.loc[adata.obs["species"] != species, "cell_type_tmp"] = (
            "other species"
        )
        cell_types = list(sorted(adata.obs["cell_type_tmp"].unique()))
        cell_types.remove("other species")
        colors = sns.color_palette("husl", n_colors=len(cell_types))
        palette = dict(zip(cell_types, colors))
        palette["other species"] = (0.5, 0.5, 0.5)
        sc.pl.umap(
            adata,
            color="cell_type_tmp",
            title=f"Cell Type ({species})",
            add_outline=True,
            size=15,
            palette=palette,
            ax=ax,
        )
        ax.legend(ncol=2, fontsize=6, loc="best")
    for ax in axs:
        ax.set(xlabel=None, ylabel=None, xticks=[], yticks=[])
    fig3.tight_layout()

    print("Closest cell type matches")
    print("10 absolute closest")
    import torch

    cdis = torch.cdist(
        torch.tensor(adata_inf.X).to("cuda"),
        torch.tensor(adata_train.X).to("cuda"),
    )
    closest = torch.topk(cdis.ravel(), 10, largest=False)
    close_idx = np.unravel_index(closest.indices.cpu().numpy(), cdis.shape)
    for i_inf, i_train in zip(*close_idx):
        print(
            f"{adata_inf.obs['ref_labels'].values[i_inf]} -> {adata_train.obs['ref_labels'].values[i_train]}"
        )

    print("10 absolute closest in UMAP space")
    import torch

    umap_inf = adata[adata.obs["species"] == f"{args.species}-inf"].obsm["X_umap"]
    umap_train = adata[adata.obs["species"] != f"{args.species}-inf"].obsm["X_umap"]
    cdis = torch.cdist(
        torch.tensor(umap_inf).to("cuda"),
        torch.tensor(umap_train).to("cuda"),
    )
    closest_umap = torch.topk(cdis.ravel(), 10, largest=False)
    close_idx = np.unravel_index(closest_umap.indices.cpu().numpy(), cdis.shape)
    for i_inf, i_train in zip(*close_idx):
        print(
            f"{adata_inf.obs['ref_labels'].values[i_inf]} -> {adata_train.obs['ref_labels'].values[i_train]}"
        )

    print("By cell type")
    from collections import Counter

    closest = cdis.min(axis=1)
    close_dic = Counter()
    for i_inf, i_train in enumerate(closest.indices.cpu().numpy()):
        close_dic[
            (
                adata_inf.obs["ref_labels"].values[i_inf],
                adata_train.obs["ref_labels"].values[i_train],
            )
        ] += 1
    close_dic = pd.Series(close_dic)
    close_dic = close_dic.unstack(fill_value=0)
    close_dic_frac = (1.0 * close_dic.T / close_dic.sum(axis=1)).T
    for idx, row in close_dic.iterrows():
        print(idx)
        for ct, count in row.nlargest(3).items():
            if count == 0:
                break
            pct = int((100 * close_dic_frac.loc[idx, ct]).round(0))
            print(f"  {ct}: {count} ({pct}%)")

    print(
        "Get percentages of correct predictions, whenever the correct cell type is found in the guide species"
    )
    # NOTE: this is somewhat unfair towards rare cell types because they can fall into the attraction basin of much bigger types
    # TODO: set up a statistically fairER test below
    from collections import defaultdict

    cell_types_guide = adata_train.obs["ref_labels"].cat.categories
    score = defaultdict(list)
    for idx, row in close_dic.iterrows():
        if idx not in cell_types_guide:
            score["missing_from_guide"].append(idx)
        else:
            best_match = row.idxmax()
            if best_match == idx:
                score["correct"].append(idx)
            else:
                score["incorrect"].append((idx, best_match))
    print(score)
    print(
        f"Correct: {len(score['correct'])}, incorrect: {len(score['incorrect'])}, missing from guide: {len(score['missing_from_guide'])}"
    )
