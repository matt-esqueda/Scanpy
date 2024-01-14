#!/usr/bin/env python3

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as pl 
from matplotlib import rcParams
import scanpy as sc 

sc.settings.verbosity = 3 
sc.logging.print_versions()
results_file = './write/paul15.h5ad'
sc.settings.set_figure_params(dpi=80, frameon=False, figsize=(3,3), facecolor='white')                              # low dpi yields small inline figures

adata = sc.datasets.paul15()
adata

adata.X = adata.X.astype('float64')                                                                                 # not required                                                                   


### Preprocessing and Visualization ###
# preprocessing recipe https://scanpy.readthedocs.io/en/latest/generated/scanpy.api.pp.recipe_zheng17.html
sc.pp.recipe_zheng17(adata)

sc.tl.pca(adata, svd_solver='arpack')

sc.pp.neighbors(adata, n_neighbors=4, n_pcs=20)
sc.tl.draw_graph(adata)

sc.pl.draw_graph(adata, color='paul15_clusters', legend_loc='on data')


### Denoising the graph (optional) ###
sc.tl.diffmap(adata)
sc.pp.neighbors(adata, n_neighbors=10, use_rep='X_diffmap')

sc.tl.draw_graph(adata)

sc.pl.draw_graph(adata, color='paul15_clusters', legend_loc='on data')


### Clustering and PAGA ###
sc.tl.louvain(adata, resolution=0.1)

# PAGA graph
sc.tl.paga(adata, groups='louvain')                                                                                 # use sc.tl.leiden instead of sc.tl.louvain

sc.pl.paga(adata, color=['louvain', 'Hba-a2', 'Elane', 'Irf8'])
sc.pl.paga(adata, color=['louvain', 'Itgab2', 'Prss34', 'Cma1'])


# annotate the clusters 
adata.obs['louvain'].cat.categories

adata.obs['louvain_anno'] = adata.obs['louvain']

# use the annotated clusters for PAGA
sc.tl.paga(adata, groups='louvain_anno')
sc.pl.paga(adata, threshold=0.03, show=False)


### Recomputing the embedding using PAGA-initialization ###
sc.tl.draw_graph(adata, init_pos='paga')

sc.pl.draw_graph(adata, color=['louvain_anno', 'Itga2b', 'Prss34', 'Cma1'], legend_loc='on data')

pl.figure(figsize=(8, 2))
for i in range(28):
    pl.scatter(i, 1, c=sc.pl.palettes.zeileis_28[i], s=200)
pl.show()

zeileis_colors = np.array(sc.pl.palettes.zeileis_28)
new_colors = np.array(adata.uns['louvain_anno_colors'])

adata.uns['louvain_anno_colors'] = new_colors

sc.pl.paga_compare(
    adata, threshold=0.03, title='', right_margin=0.2, size=10, edge_width_scale=0.5,
    legend_fontsize=12, fontsize=12, frameon=False, edges=True, save=True)


### Reconstructing gene changes along PAGA paths for a given set of genes ###
adata.uns['iroot'] = np.flatnonzero(adata.obs['louvain_anno']  == '16')[0]

sc.tl.dpt(adata)

gene_names = ['Gata2', 'Gata1', 'Klf1', 'Epor', 'Hba-a2',  # erythroid
              'Elane', 'Cebpe', 'Gfi1',                    # neutrophil
              'Irf8', 'Csf1r', 'Ctsg']                     # monocyte

sc.pl.draw_graph(adata, color=['louvain_anno', 'dpt_pseudotime'], legend_loc='on data')

paths = [('erythrocytes', [16, 12, 7, 13, 18, 6, 5, 10]),
         ('neutrophils', [16, 0, 4, 2, 14, 19]),
         ('monocytes', [16, 0, 4, 11, 1, 9, 24])]

adata.obs['distance'] = adata.obs['dpt_pseudotime'] 
adata.obs['clusters'] = adata.obs['louvain_anno']
adata.uns['clusters_colors'] = adata.uns['louvain_anno_colors']