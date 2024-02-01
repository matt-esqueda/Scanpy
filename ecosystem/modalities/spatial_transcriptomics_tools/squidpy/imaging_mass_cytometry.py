#!/usr/bin/env python3

### Import packages and data ###
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import squidpy as sq

print(f"squidpy=={sq.__version__}")

# Load the pre-processed dataset
adata = sq.datasets.imc()

# viusalize the cluster annotation in spatial context
sq.pl.spatial_scatter(adata, shape=None, color="cell type", size=10)


### Co-occurrence across spatial dimensions ###
sq.gr.co_occurrence(adata, cluster_key="cell type")
sq.pl.co_occurrence(
    adata,
    cluster_key="cell type",
    clusters=["basal CK tumor cell", "T cells"],
    figsize=(15, 4),
)


### Neighborhood enrichment ###
# compute a connectivity matrix and visualize the results
sq.gr.spatial_neighbors(adata)
sq.gr.nhood_enrichment(adata, cluster_key="cell type")
sq.pl.nhood_enrichment(adata, cluster_key="cell type")


### Interaction matrix and network centralities ###
# compute an interaction matrix
sq.gr.interaction_matrix(adata, cluster_key="cell type")
sq.pl.interaction_matrix(adata, cluster_key="cell type")

# investigate properties of the spatial graph by computing different network centralities
sq.gr.centrality_scores(
    adata,
    cluster_key="cell type",
)
sq.pl.centrality_scores(adata, cluster_key="cell type", figsize=(20, 5), s=500)
