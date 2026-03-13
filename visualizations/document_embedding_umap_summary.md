# Document Embedding UMAP

- Documents projected: 7,129
- Embedded pages processed: 171,860
- Sampled page points shown in background: 8,000
- Rows skipped due to malformed/missing embeddings: 0
- Median pages per document: 5.0
- 90th percentile pages per document: 48.0
- UMAP metric: cosine
- UMAP n_neighbors: 25
- UMAP min_dist: 0.08

Interpretation:
- Each colored point is one document centroid computed as the mean of its L2-normalized page embeddings.
- UMAP emphasizes local neighborhoods, so nearby points are usually more meaningful than large global distances.
- Gaps and island-like clusters can indicate real local structure, but axis values themselves have no direct semantic meaning.
- The gray points are sampled pages transformed with the same fitted UMAP model.
