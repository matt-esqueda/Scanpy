#!/usr/bin/env python3

import scvelo as scv
scv.logging.print_version()

scv.settings.verbosity = 3
scv.settings.presenter_view = True
scv.settings.set_figure_params('scvelo')


# prepare the data
adata = scv.datasets.dentategyrus()

scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=2000)
scv.pp.moments(adata, n_pcs=30, n_neighbors=30)


# basic velocity estimation
scv.tl.velocity(adata)
scv.tl.velocity_graph(adata)

scv.pl.velocity_embedding_stream(adata, basis='umap')


# differential kinetics test
var_names = ['Tmsb10', 'Fam155a', 'Hn1', 'Rpl6']
scv.tl.differential_kinetic_test(adata, var_names=var_names, groupby='clusters')

scv.get_df(adata[:, var_names], ['fit_diff_kinetics', 'fit_pval_kinetics'], precision=2)

kwargs = dict(linewidth=2, add_linfit=True, frameon=False)
scv.pl.scatter(adata, basis=var_names, add_outline='fit_diff_kinetics', **kwargs)

diff_clusters = list(adata[:, var_names].var['fit_diff_kinetics'])
scv.pl.scatter(adata, legend_loc='right', size=60, title='diff kinetics',
              add_outline=diff_clusters,outline_width=(.8, .2))


# testing toplikelihood genes
scv.tl.recover_dynamics(adata)

# adata.write('data/pancreas.h5ad', compression='gzip')
# adata = scv.read('data/pancreas.h5ad')

top_genes = adata.var['fit_likelihood'].sort_values(ascending=False).index[:100]
scv.tl.differential_kinetic_test(adata, var_names=top_genes, groupby='clusters')

scv.pl.scatter(adata, basis=top_genes[:15], ncols=5, add_outline='fit_diff_kinetics', **kwargs)

scv.pl.scatter(adata, basis=top_genes[15:30], ncols=5, add_outline='fit_diff_kinetics', **kwargs)


# recompute velocities
scv.tl.velocity(adata, diff_kinetics=True)
scv.tl.velocity_graph(adata)

scv.pl.velocity_embedding(adata, dpi=120, arrow_size=2, arrow_length=2)