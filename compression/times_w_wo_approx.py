import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import atlasapprox


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--savefig", action="store_true", help="Save the figures")
    args = parser.parse_args()

    print("Time estimates")
    timesd = {
        # name: [w/o (+loading), w/o (no loading), w/]
        "plot umap for 10 genes": [60, 30, 1.7],
        "find and plot markers": [60, 30, 1.2],
        "find highest expressor across tissues": [15 * 60, 15 * 60, 1.2],
        "find similar genes": [6 * 60 + 30, 6 * 60, 1.2],
        "find sequence of 10 genes": [30 * 60, 30 * 60, 1.2],
        "check what cell types are found in what tissue": [10 * 60, 10 * 60, 1.2],
        "accessing atlas the first time": [3 * 60 * 60, 3 * (60 + 5) * 60, 1.2],
        "comparing the same cell type across tissues": [15 * 60, 15 * 60, 1.2],
        "find the sequence of markers for the same cell type in mouse and human": [
            2 * 60 * 60,
            2 * 60 * 60,
            2.45,
        ],
        "check cell type abundance across embedding": [3.5 * 60, 3 * 60, 5],
        "check accessibility of a peak in a cell type": [10 * 60, 60, 1.4],
        "check what organisms have a cell atlas": [5 * 24 * 60 * 60, 8 * 60 * 60, 1.2],
    }

    print("Check a few estimates")
    plt.ioff()

    print(" Plot UMAP for 10 genes")
    genes = [
        "CD3E",
        "CD4",
        "CD8A",
        "CD19",
        "CD14",
        "PTPRC",
        "NCAM1",
        "COL1A1",
        "CD34",
        "CD44",
    ]
    t0 = time.time()
    api = atlasapprox.API()
    res = api.neighborhood(
        "h_sapiens",
        "lung",
        features=genes,
    )
    avlog = np.log1p(res["average"])
    xmax, ymax = pd.concat(res["boundaries"]).max(axis=0).values * 1.05
    xmin, ymin = pd.concat(res["boundaries"]).min(axis=0).values * 1.05
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    axs = axs.ravel()
    for ax, gene in zip(axs, genes):
        emax = avlog.loc[gene].max()
        for i, hull in enumerate(res["boundaries"]):
            ge = avlog.at[gene, i]
            color = plt.cm.viridis(ge / emax)
            ax.add_patch(
                plt.Polygon(
                    hull.values,
                    facecolor=color,
                    edgecolor="k",
                    lw=1.5,
                )
            )
            xc, yc = res["centroids"].loc[i].values
            ax.text(xc, yc, i, ha="center", va="center", fontsize=6)
        ax.set_title(gene)
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    fig.tight_layout()
    t1 = time.time()
    print("  Time estimate:", t1 - t0, "s")
    timesd["plot umap for 10 genes"][-1] = t1 - t0

    print(" Find and plot markers")
    t0 = time.time()
    markers = api.markers("h_sapiens", "lung", "T", number=5)
    res = atlasapprox.pl.dotplot(api, "h_sapiens", "lung", markers)
    t1 = time.time()
    print("  Time estimate:", t1 - t0, "s")
    timesd["find and plot markers"][-1] = t1 - t0

    print(" Find sequence of 10 genes")
    genes = [
        "CD3E",
        "CD4",
        "CD8A",
        "CD19",
        "CD14",
        "PTPRC",
        "NCAM1",
        "COL1A1",
        "CD34",
        "CD44",
    ]
    t0 = time.time()
    api = atlasapprox.API()
    sequences = api.sequences("h_sapiens", genes)["sequences"]
    t1 = time.time()
    print(" Time estimate:", t1 - t0, "s")
    timesd["find sequence of 10 genes"][-1] = t1 - t0

    print(" Check what cell types are found in what tissue")
    t0 = time.time()
    api = atlasapprox.API()
    res = api.celltypexorgan("h_sapiens")
    t1 = time.time()
    print(" Time estimate:", t1 - t0, "s")
    timesd["check what cell types are found in what tissue"][-1] = t1 - t0

    print(" Find highest expressor across tissues")
    t0 = time.time()
    api = atlasapprox.API()
    res = api.highest_measurement(
        "h_sapiens",
        "CD14",
        number=10,
    )
    t1 = time.time()
    print(" Time estimate:", t1 - t0, "s")
    timesd["find highest expressor across tissues"][-1] = t1 - t0

    print("  Find the sequence of markers for the same cell type in mouse and human")
    ct = "NK"
    species = ["h_sapiens", "m_musculus"]
    organ = "lung"
    t0 = time.time()
    api = atlasapprox.API()
    resd = {}
    for spec in species:
        markers = api.markers(spec, organ, ct, number=5)
        sequences = api.sequences(spec, markers)["sequences"]
        resd[spec] = dict(zip(markers, sequences))
    t1 = time.time()
    print("  Time estimate:", t1 - t0, "s")
    timesd["find the sequence of markers for the same cell type in mouse and human"][
        -1
    ] = (t1 - t0)

    print(" Check what organisms have a cell atlas")
    t0 = time.time()
    api = atlasapprox.API()
    res = api.organisms()
    t1 = time.time()
    print(" Time estimate:", t1 - t0, "s")
    timesd["check what organisms have a cell atlas"][-1] = t1 - t0

    if False:
        print("Plot overall statistics, verbose")
        x1, x2, y = np.array(list(timesd.values())).T
        xs = [x1, x2]
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(3, 3.7))
        for i, ax in enumerate(axs):
            ax.scatter(xs[i], y, s=20, color="k")
            ax.grid(True)
            ax.set_xscale("log")
            ratio = int((xs[i] / y).mean())
            ax.text(
                0.05,
                0.9,
                "$\\eta \\approx$" + str(ratio),
                va="top",
                ha="left",
                transform=ax.transAxes,
                bbox=dict(facecolor="white"),
            )
        axs[0].set_xlabel("Time w/o approximations\n(inc. loading) [s]")
        axs[1].set_xlabel("Time w/o approximations\n(exc. loading) [s]")
        axs[0].set_ylabel("Time with\napproximations [s]")
        axs[1].set_ylabel("Time with\napproximations [s]")
        fig.tight_layout()

    print("Plot overall statistics, compact")
    fig, ax = plt.subplots(figsize=(3.3, 2))
    x, _, y = np.array(list(timesd.values())).T
    ax.scatter(x, y, s=25, color=[0.1, 0.1, 0.1], zorder=10)
    ax.grid(True)
    ax.set_xscale("log")
    ratio = int((x / y).mean())
    ax.text(
        0.95,
        0.9,
        "$\\eta \\approx$" + str(ratio),
        va="top",
        ha="right",
        transform=ax.transAxes,
        bbox=dict(facecolor="white"),
    )
    ax.set_xlabel("Traditional analysis\nestimated runtime [s]")
    ax.set_ylabel("API runtime [s]")
    fig.tight_layout()

    plt.ion()
    plt.show()

    if args.savefig:
        fig.savefig("../figures/runtimes_estimate.svg")
        fig.savefig("../figures/runtimes_estimate.png", dpi=300)
