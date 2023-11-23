<img src="https://raw.githubusercontent.com/fabilab/cell_atlas_approximations/main/figures/figure_Approximation.png" width="150">

# Cell Atlas Approximations - Compression
Cell atlases such as Tabula Muris and Tabula Sapiens are multi-organ single cell omics data sets describing entire organisms. A cell atlas approximation is a lossy and lightweight compression of a cell atlas that can be streamed via the internet.

This project enables biologists, doctors, and data scientist to quickly find answers for questions such as:

- *What types of cells populate the human heart?*
- *What is the expression of a specific gene across cell types in C elegans?*
- *What are the marker genes of a specific cell type in mouse pancreas*?
- *What fraction of cells (of a specific type) express a gene of interest?*

To answer these questions, we construct approximations from all kinds of cell atlases using the code in this repo.

**NOTE:** At this stage, this repo is designed for internal use and made public for transparency. End users are expected to interact with the atlas approximations using the [web interface](https://atlasapprox.org) or the [API](https://atlasapprox.readthedocs.io).

## Available approximations

| Organism | Organs | Publication |
| --- | --- | --- |
| homo sapiens | Bone marrow, heart, kidney, colon, pancreas, lung, tongue | https://www.science.org/doi/10.1126/science.abl4896 |
| mus musculus | Bone marrow, heart, kidney, colon, pancreas, lung, tongue | https://www.nature.com/articles/s41586-020-2496-1 |
| microcebus myoxinus | Bone marrow, heart, kidney, colon, pancreas, lung, tongue | https://www.biorxiv.org/content/10.1101/2021.12.12.469460v2 |
| caenorhabditis elegans | whole organism | https://www.science.org/doi/10.1126/science.aam8940 |
| drosophila melanogaster | Antenna, haltere, heart, intestines, leg, male reproductive glands, malpighian tubule, oenocyte, ovary, proboscis, skin, testis, trachea, wing | https://doi.org/10.1126/science.abk2432 |
| danio rerio | whole embryo | https://www.science.org/doi/10.1126/science.aar4362 |
| hofstenia miamia | whole organism | https://www.nature.com/articles/s41467-023-38016-4 |
| schmidtea mediterraea | whole organism | https://www.science.org/doi/10.1126/science.aaq1736 |
| schistosoma mansoni | whole organism | https://www.nature.com/articles/s41467-020-20794-w |
| spongilla lacustris | whole organism | https://www.science.org/doi/10.1126/science.abj2949 |
| stylophora pistillata | whole organism | https://www.sciencedirect.com/science/article/pii/S0092867421004402 |
| amphimedon queenslandica | whole organism | https://www.nature.com/articles/s41559-018-0575-6 |
| mnemiopsis leidy | whole organism | https://www.nature.com/articles/s41559-018-0575-6 |
| trichoplax adhaerens | whole organism | https://www.nature.com/articles/s41559-018-0575-6 |
| xenopus laevis | Bladder, bone marrow, brain, eye, heart, intestine, kidney, liver, lung, muscle, ovary, oviduct, pancreas, skin, spleen, stomach, testis | https://www.nature.com/articles/s41467-022-31949-2 |


## Proposing a new atlas approximation
If you would like to propose an additional cell atlas to be added to the list of approximations, please open an [issue](https://github.com/fabilab/cell_atlas_approximations_compression/issues/new/choose) and specify:

- a hyperlink to a publication where the atlas is described
- a hyperlink to a data source where the table of counts can be downloaded
- any information you might have about cell type annotation
- the reason why you would like to have that atlas added here

Thank you for your time.
