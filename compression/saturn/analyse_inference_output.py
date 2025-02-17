"""Analyse the output of SATURN

This script requires anndata and scanpy. One way to do that is to use the SATURN conda environment:

source ~/miniconda3/bin/activate && conda activate saturn

"""

import os
import sys
import pathlib
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


plants = [
    "t_aestivum",
    "f_vesca",
    "a_thaliana",
    "l_minuta",
    "z_mays",
]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analyse the output of SATURN.")
    parser.add_argument(
        "--n-macro", type=int, help="Number of macrogenes", required=True
    )
    parser.add_argument(
        "--n-hvg", type=int, help="Number of highly variable  genes", required=True
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of epochs in metric learning", required=True
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        help="Number of epochs in pretraining",
        required=True,
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

    print("Read trained h5ad")
    adata_train = anndata.read_h5ad(saturn_h5ad)
    adata_train.obs["train_inf"] = "train"

    print("Read inference h5ad")
    inf_h5ad = next((output_fdn / "inference_output").iterdir())
    species_guide = "_".join(inf_h5ad.stem.split("_")[2:4])
    species_inf = species_guide + "-inf"
    adata_inf = anndata.read_h5ad(inf_h5ad)
    adata_inf.obs["train_inf"] = "inference"

    print("Concatenate training and inference anndata")
    adata = anndata.concat([adata_train, adata_inf])

    print("Add tissue information")
    adata.obs["organ"] = ""
    separate_h5ad_fdn = output_fdn.parent / "h5ads"
    for species, datum in df.iterrows():
        print(species)
        h5ad_fn = pathlib.Path(datum["path"])
        if str(h5ad_fn).startswith("/srv/scratch"):
            h5ad_fn = pathlib.Path(
                f"/mnt/data/projects/cell_atlas_approximations/reference_atlases/data/saturn/h5ads/{species}.h5ad"
            )
        if not h5ad_fn.exists():
            species_orig = species.split("-")[0]
            h5ad_fn = pathlib.Path(
                f"/mnt/data/projects/cell_atlas_approximations/reference_atlases/data/saturn/h5ads/{species_orig}.h5ad"
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

    if args.kingdom == "animal":
        adata = adata[~adata.obs["species"].isin(plants)]
    elif args.kingdom == "plant":
        adata = adata[adata.obs["species"].isin(plants)]

    print("PCA")
    sc.pp.pca(adata)

    print("KNN")
    sc.pp.neighbors(adata)

    print("UMAP")
    sc.tl.umap(adata, n_components=args.umap_dim)

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

    print("Visualise")
    plt.ion()
    plt.close("all")

    if args.umap_dim == 2:

        if True:
            sc.pl.umap(
                adata, color="species", title="Species", add_outline=True, size=20
            )
            fig = plt.gcf()
            fig.set_size_inches(9, 5)
            fig.tight_layout()
            # fig.savefig("../figures/combined_umap_saturn_atlasapprox_species.png", dpi=300)

            # Same but only for guide and inf species
            palette = {key: "grey" for key in adata.obs["species"].cat.categories}
            palette[species_inf] = "tomato"
            palette[species_guide] = "steelblue"
            sc.pl.umap(
                adata,
                color="species",
                title="Species",
                groups=[species_inf, species_guide],
                palette=palette,
                add_outline=True,
                size=20,
            )
            fig = plt.gcf()
            fig.set_size_inches(8, 5)
            fig.tight_layout()

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

    _import__("sys").exit()

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
