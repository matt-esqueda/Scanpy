#!/usr/bin/env python3

import scvelo as scv
import pandas as pd
scv.logging.print_version()

scv.settings.verbosity=3 
scv.settings.presenter_view = True        
scv.set_figure_params('scvelo')


# load the data
adata = scv.datasets.pancreas()
print(adata)

scv.pl.proportions(adata)


# preprocess the data
scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
scv.pp.moments(adata, n_pcs=30, n_neighbors=30)


# estimate RNA velocity
scv.tl.velocity(adata)
scv.tl.velocity_graph(adata)


# project the velocities
scv.pl.velocity_embedding_stream(adata, basis='umap')

scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, dpi=120)


# interprete the velocities
scv.pl.velocity(adata, ['Cpe', 'Gnao1', 'Ins2', 'Adk'], ncols=2)

scv.pl.scatter(adata, 'Cpe', color=['clusters', 'velocity'],
              add_outline='Ngn3 high EP, Pre-endocrine, Beta')


# identify important genes
scv.tl.rank_velocity_genes(adata, groupby='clusters', min_corr=.3)

df = pd.DataFrame(adata.uns['rank_velocity_genes']['names'])
print(df.head())

kwargs = dict(frameon=False, size=10, linewidth=1.5,
              add_outline='Ngn3 high EP, Pre-endocrine, Beta')

scv.pl.scatter(adata, df['Ngn3 high EP'][:5], ylabel='Ngn3 high EP', **kwargs)
scv.pl.scatter(adata, df['Pre-endocrine'][:5], ylabel='Pre-endocrine', **kwargs)


# velocities in cycling progenitors
scv.tl.score_genes_cell_cycle(adata)
scv.pl.scatter(adata, color_gradients=['S_score', 'G2M_score'], smooth=True, perc=[5, 95])

s_genes, g2m_genes = scv.utils.get_phase_marker_genes(adata)
s_genes = scv.get_df(adata[:, s_genes], 'spearmans_score', sort_values=True).index
g2m_genes = scv.get_df(adata[:, g2m_genes], 'spearmans_score', sort_values=True).index

kwargs = dict(frameon=False, ylabel='cell cycle genes')
scv.pl.scatter(adata, list(s_genes[:2]) + list(g2m_genes[:3]), **kwargs)

scv.pl.velocity(adata, ['Hells', 'Top2a'], ncols=2, add_outline=True)


# speed and coherence
scv.tl.velocity_confidence(adata)
keys = 'velocity_length', 'velocity_confidence'
scv.pl.scatter(adata, c=keys, cmap='coolwarm', perc=[5, 95])

df = adata.obs.groupby('clusters')[['velocity_length', 'velocity_confidence']].mean().T
df.style.background_gradient(cmap='coolwarm', axis=1)


# velocity graph and pseudotime
scv.pl.velocity_graph(adata, threshold=.1)

x, y = scv.utils.get_cell_transitions(adata, basis='umap', starting_cell=70)
ax = scv.pl.velocity_graph(adata, c='lightgrey', edge_width=.05, show=False)
ax = scv.pl.scatter(adata, x=x, y=y, s=120, c='ascending', cmap='gnuplot', ax=ax)

scv.tl.velocity_pseudotime(adata)
scv.pl.scatter(adata, color='velocity_pseudotime', cmap='gnuplot')


# PAGA velocity graph
adata.uns['neighbors']['distances'] = adata.obsp['distances']
adata.uns['neighbors']['connectivities'] = adata.obsp['connectivities']

scv.tl.paga(adata, groups='clusters')
df = scv.get_df(adata, 'paga/transitions_confidence', precision=2).T
df.style.background_gradient(cmap='Blues').format('{:.2g}')

scv.pl.paga(adata, basis='umap', size=50, alpha=.1,
           min_edge_width=2, node_size_scale=1.5)