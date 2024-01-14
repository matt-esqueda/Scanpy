#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

### Load the data ###
adata_pbmc3k = sc.read_10x_mtx("tutorial_data/pbmc3k_v1/", cache=True)
adata_pbmc10k = sc.read_10x_mtx("tutorial_data/pbmc10k_v3/", cache=True)

adata_pbmc3k.uns["name"] = "PBMC 3k (v1)"
adata_pbmc10k.uns["name"] = "PBMC 10k (v3)"

# marker genes from table in pbmc3k tutorial
markers = [
    "IL7R",
    "LYZ",
    "CD14",
    "MS4A1",
    "CD8A",
    "GNLY",
    "NKG7",
    "FCGR3A",
    "MS4A7",
    "FCER1A",
    "CST3",
    "PPBP",
]


### Perform quality control ###
# basic filtering
for adata in [adata_pbmc3k, adata_pbmc10k]:
    adata.var_names_make_unique()
    print(adata.uns["name"], ": data shape:", adata.shape)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

# compute QC metrics
for adata in [adata_pbmc3k, adata_pbmc10k]:
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

# plot QC metrics
for adata in [adata_pbmc3k, adata_pbmc10k]:
    print(adata.uns["name"], ":")
    sc.pl.violin(
        adata,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        jitter=0.4,
        multi_panel=True,
    )

# define outliers and do the filtering for the 3k dataset
adata_pbmc3k.obs['outlier_mt'] = adata_pbmc3k.obs.pct_counts_mt > 5
adata_pbmc3k.obs['outlier_total'] = adata_pbmc3k.obs.total_counts > 5000
adata_pbmc3k.obs['outlier_genes'] = adata_pbmc3k.obs.n_genes_by_counts > 25000

print('%u cells with high %% of mitochondrial genes' % (sum(adata_pbmc3k.obs['outlier_mt'])))
print('%u cells with large total counts' % (sum(adata_pbmc3k.obs['outlier_total'])))
print('%u cells with large number of genes' % (sum(adata_pbmc3k.obs['outlier_genes'])))

adata_pbmc3k = adata_pbmc3k[~adata_pbmc3k.obs['outlier_mt'], :]
adata_pbmc3k = adata_pbmc3k[~adata_pbmc3k.obs['outlier_total'], :]
adata_pbmc3k = adata_pbmc3k[~adata_pbmc3k.obs['outlier_genes'], :]
sc.pp.filter_genes(adata_pbmc3k, min_cells=1)

# define outliers and do the filtering for the 10k dataset
adata_pbmc10k.obs['outlier_mt'] = adata_pbmc10k.obs.pct_counts_mt > 20
adata_pbmc10k.obs['outlier_total'] = adata_pbmc10k.obs.total_counts > 25000
adata_pbmc10k.obs['outlier_ngenes'] = adata_pbmc10k.obs.n_genes_by_counts > 6000

print('%u cells with high %% of mitochondrial genes' % (sum(adata_pbmc10k.obs['outlier_mt'])))
print('%u cells with large total counts' % (sum(adata_pbmc10k.obs['outlier_total'])))
print('%u cells with large number of genes' % (sum(adata_pbmc10k.obs['outlier_ngenes'])))

adata_pbmc10k = adata_pbmc10k[~adata_pbmc10k.obs['outlier_mt'], :]
adata_pbmc10k = adata_pbmc10k[~adata_pbmc10k.obs['outlier_total'], :]
adata_pbmc10k = adata_pbmc10k[~adata_pbmc10k.obs['outlier_ngenes'], :]
sc.pp.filter_genes(adata_pbmc10k, min_cells=1)


### Use Pearson residuals for selction of highly variable genes ###
# compute 2000 variable genes with Pearson residuals
for adata in [adata_pbmc3k, adata_pbmc10k]:
    sc.experimental.pp.highly_variable_genes(
        adata, flavor="pearson_residuals", n_top_genes=2000
    )

# plot gene selection
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, adata in zip(axes, [adata_pbmc3k, adata_pbmc10k]):
    
    hvgs = adata.var["highly_variable"]
    
    ax.scatter(
        adata.var["mean_counts"], adata.var["residual_variances"], s=3, edgecolor="none"
    )
    ax.scatter(
        adata.var["mean_counts"][hvgs],
        adata.var["residual_variances"][hvgs],
        c="tab:red",
        label="selected genes",
        s=3,
        edgecolor="none",
    )
    ax.scatter(
        adata.var["mean_counts"][np.isin(adata.var_names, markers)],
        adata.var["residual_variances"][np.isin(adata.var_names, markers)],
        c="k",
        label="known marker genes",
        s=10,
        edgecolor="none",
    )
    ax.set_xscale("log")
    ax.set_xlabel("mean expression")
    ax.set_yscale("log")
    ax.set_ylabel("residual variance")
    ax.set_title(adata.uns["name"])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
plt.legend()

# apply gene selection
adata_pbmc3k = adata_pbmc3k[:, adata_pbmc3k.var["highly_variable"]]
adata_pbmc10k = adata_pbmc10k[:, adata_pbmc10k.var["highly_variable"]]

# print resulting adata objects
print(adata_pbmc3k)
print(adata_pbmc10k)


### Transforming raw counts to Pearson residuals ###
# keep raw and depth-normalized counts for later
adata_pbmc3k.layers["raw"] = adata_pbmc3k.X.copy()
adata_pbmc3k.layers["sqrt_norm"] = np.sqrt(
    sc.pp.normalize_total(adata_pbmc3k, inplace=False)["X"]
)

adata_pbmc10k.layers["raw"] = adata_pbmc10k.X.copy()
adata_pbmc10k.layers["sqrt_norm"] = np.sqrt(
    sc.pp.normalize_total(adata_pbmc10k, inplace=False)["X"]
)

# compute Pearson residuals
for adata in [adata_pbmc3k, adata_pbmc10k]:
    sc.experimental.pp.normalize_pearson_residuals(adata)

# compute PCA and t-SNE
for adata in [adata_pbmc3k, adata_pbmc10k]:
    sc.pp.pca(adata, n_comps=50)
    n_cells = len(adata)
    sc.tl.tsne(adata, use_rep="X_pca")

# compute neighborhood graph and Leiden clustering
for adata in [adata_pbmc3k, adata_pbmc10k]:
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
    sc.tl.leiden(adata)

# plot Leiden clusters on tSNE and PBMC marker genes
for adata in [adata_pbmc3k, adata_pbmc10k]:
    print(adata.uns["name"], ":")
    sc.pl.tsne(adata, color=["leiden"], cmap="tab20")
    sc.pl.tsne(adata, color=markers, layer="sqrt_norm")