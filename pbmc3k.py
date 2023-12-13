#!/usr/bin/env python3

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np 
import pandas as pd 
import scanpy as sc 

sc.settings.verbosity = 3                                                       # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')

results_file = './write/pbmc3k.h5ad'                                            # the file that will store the analysis results

# read in the count matrix into an AnnData object
adata = sc.read_10x_mtx(
    'data/filtered_gene_bc_matrices/hg19/',                                     # the directory with the '.mtx' file
    var_names='gene_symbols',                                                   # use gene symbols for the variable name (variables-axis index)
    cache=True)                                                                 # write a cache file for faster subsequent reading

adata.var_names_make_unique()                                                   # this is unecessary if using `var_names='gene_ids'` in `sc.read_10x_mtx`
adata


### Preprocessing ###
# show the genes that yield the highest fraction of counts in each single cell, across all cells
sc.pl.highest_expr_genes(adata, n_top=20, )

# basic filtering 
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

adata.var['mt'] = adata.var_names.str.startswith('MT-')                         # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], 
    percent_top=None, log1p=False, inplace=True)

# violin plots of computed quality metrics
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
            jitter=0.4, multi_panel=True)

# remove cells with too many mt genes expressed or too many total counts
sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')

# filter by slicing AnnData obj
adata = adata[adata.obs.n_genes_by_counts < 2500, :]
adata = adata[adata.obs.pct_counts_mt < 5, :]

# normalize the data matrix to 10,000 reads per cell so that counts are compareable among cells
sc.pp.normalize_total(adata, target_sum=1e4)

# logarithmize the data
sc.pp.log1p(adata)

# identify highly-variable genes
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pl.highly_variable_genes(adata)

# set the .raw attibute of the AnnData obj to the norm/log raw gene expression 
adata.raw = adata                                                               # can get back AnnData of raw obj by calling .raw.to_adata()

# do the filterinng on adata
adata = adata[:, adata.var.highly_variable]
sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
sc.pp.scale(adata, max_value=10)


### Principal component analysis ###
# reduce the dimensionaluty of the data
sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca(adata, color='CST3')

# inspect the contribution of single PCs to the total variance
sc.pl.pca_variance_ratio(adata, log=True)

adata.write(results_file)
adata


### Computing the neighborhood graph ###
# compute neighborhood graph of cells using the PCA representation 
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)


### Embedding the neighborhood graph ###
# sc.tl.paga(adata)
# sc.pl.paga(adata, plot=False)                                                      
# sc.tl.umap(adata, init_pos='paga')

sc.tl.umap(adata)
# with "raw" (norm, log, but uncorrected) data
sc.pl.umap(adata, color=['CST3', 'NKG7', 'PPBP'])
# with scaled and corrected gene expression
sc.pl.umap(adata, color=['CST3', 'NKG7', 'PPBP'], use_raw=False)


### Clustering the neighborhood graph ###
# community based on optimizing modularity
sc.tl.leiden(adata)
# plot the clusters
sc.pl.umap(adata, color=['leiden', 'CST3', 'NKG7'])
adata.write(results_file)


### Finding marker genes ###
# t-test to compute a ranking for highly differntial genes in each cluster
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)

sc.settings.verbosity = 2                                                       # reduce the verbosity

# Wilcoxon rank-sum ranking
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)

adata.write(results_file)

# rank genes using logistic regression
sc.tl.rank_genes_groups(adata, 'leiden', method='logreg')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)

# define list of marker genes
marker_genes = ['IL7R', 'CD79A', 'MS4A1', 'CD8A', 'CD8B', 'LYZ', 'CD14',
                'LGALS3', 'S100A8', 'GNLY', 'NKG7', 'KLRB1',
                'FCGR3A', 'MS4A7', 'FCER1A', 'CST3', 'PPBP']

adata = sc.read(results_file)

pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(5)

# table showing the scores and groups
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
pd.DataFrame(
    {group + '_' + key[:1]: result[key][group]
    for group in groups for key in ['names', 'pvals']}).head(5)

# compare to a single cluster
sc.tl.rank_genes_groups(adata, 'leiden', group=['0'], reference='1', method='wilcoxon')
sc.pl.rank_genes_groups(adata, groups=['0'], n_genes=20)

# more detailed violin plot
sc.pl.rank_genes_groups_violin(adata, groups='0', n_genes=8)

# reload the obj with the computed differential expression
adata = sc.read(results_file)
sc.pl.rank_genes_groups_violin(adata, groups='0', n_genes=8)

# compare a gene across groups
sc.pl.violin(adata, ['CST3', 'NKG7', 'PPBP'], groupby='leiden')

# mark the cell types
new_cluster_names = [
    'CD4 T', 'CD14 Monocytes', 'B', 'CD8 T',
    'NK', 'FCGR3A Monocytes', 'Dendritic', 'Megakaryocytes']
adata.rename_categories('leiden', new_cluster_names)

sc.pl.umap(adata, color='leiden', legend_loc='on data', title='', frameon=False, save='.pdf')

# dotplot to visualize the marker genes
sc.pl.dotplot(adata, marker_genes, groupby='leiden');

# compact violin plot for marker genes
sc.pl.stacked_violin(adata, marker_genes, groupby='leiden', rotation=90);

# AnnData annotations
adata

adata.write(results_file, compression='gzip')                                       # `compression='gzip'` saves disk space, slows read/write 

# reduced file size with removed dense scaled and corrected data matrix
adata.raw.to_adata().write('./write/pbm3k_withoutX.h5ad')

