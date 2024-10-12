#!/usr/bin/Rscript       

#' Functions for converting between different objects
#' Thanks to: https://github.com/cellgeni/sceasy/blob/master/R/functions.R

#' Regularise dataframe
#'
#' This function checks if certain columns of a dataframe is of a single value
#' and drop them if required
#'
#' @param df Input data frame, usually cell metadata table (data.frame-like
#'   object)
#' @param drop_single_values Drop columns with only a single value (logical)
#'
#' @return Dataframe
.regularise_df <- function(df, drop_single_values = TRUE) {
  if (ncol(df) == 0) df[["name"]] <- rownames(df)
  if (drop_single_values) {
    k_singular <- sapply(df, function(x) length(unique(x)) == 1)
    if (sum(k_singular) > 0) {
      warning(
        paste("Dropping single category variables:"),
        paste(colnames(df)[k_singular], collapse = ", ")
      )
    }
    df <- df[, !k_singular, drop = F]
    if (ncol(df) == 0) df[["name"]] <- rownames(df)
  }
  return(df)
}

#' Convert Seurat object to AnnData
#'
#' This function converts a Seurat object to an Anndata object
#'
#' @param obj Input Seurat object
#' @param outFile Save output AnnData to this file if specified (str or NULL)
#' @param assay Assay to be converted, default "RNA" (str)
#' @param main_layer Slot in `assay` to be converted to AnnData.X, may be
#'   "counts", "data", "scale.data", default "data" (str)
#' @param transfer_layers If specified, convert slots to AnnData.layers[<slot>],
#'   (vector of str)
#' @param drop_single_values Drop single value columns in cell metadata table,
#'   default TRUE (logical)
#'
#' @return AnnData object
#'
#' @import reticulate
#' @import Matrix
seurat2anndata <- function(obj, outFile = NULL, assay = "RNA", main_layer = "data", transfer_layers = NULL, drop_single_values = TRUE) {
  if (!requireNamespace("Seurat")) {
    stop("This function requires the 'Seurat' package.")
  }
  main_layer <- match.arg(main_layer, c("data", "counts", "scale.data"))
  transfer_layers <- transfer_layers[
    transfer_layers %in% c("data", "counts", "scale.data")
  ]
  transfer_layers <- transfer_layers[transfer_layers != main_layer]

  if (compareVersion(as.character(obj@version), "3.0.0") < 0) {
    obj <- Seurat::UpdateSeuratObject(object = obj)
  }

  X <- Seurat::GetAssayData(object = obj, assay = assay, layer = main_layer)

  obs <- .regularise_df(obj@meta.data, drop_single_values = drop_single_values)

  var <- .regularise_df(Seurat::GetAssay(obj, assay = assay)@meta.features, drop_single_values = drop_single_values)

  obsm <- NULL
  reductions <- names(obj@reductions)
  if (length(reductions) > 0) {
    obsm <- sapply(
      reductions,
      function(name) as.matrix(Seurat::Embeddings(obj, reduction = name)),
      simplify = FALSE
    )
    names(obsm) <- paste0("X_", tolower(names(obj@reductions)))
  }

  layers <- list()
  for (layer in transfer_layers) {
    mat <- Seurat::GetAssayData(object = obj, assay = assay, layer = layer)
    if (all(dim(mat) == dim(X))) layers[[layer]] <- Matrix::t(mat)
  }

  anndata <- reticulate::import("anndata", convert = FALSE)

  adata <- anndata$AnnData(
    X = Matrix::t(X),
    obs = obs,
    var = var,
    obsm = obsm,
    layers = layers
  )

  if (!is.null(outFile)) {
    adata$write(outFile, compression = "gzip")
  }

  adata
}

######################################################################

argv = commandArgs(trailingOnly=TRUE)
if (is.null(argv) | length(argv)<1) {
  cat("Usage: convert_seurat_to_anndata_sceasy.R myfile.rds\n")
  q()
}

# Load Seurat
library("Seurat")

# Read RDS object (should be a Seurat object)
data <- readRDS(argv[1])

paste("Seurat object loaded")
paste(summary(data))

paste("Update Seurat object")
data = Seurat::UpdateSeuratObject(object = data)

paste("Converting Seurat object to AnnData object into", argv[2])
seurat2anndata(data, outFile = argv[2])
