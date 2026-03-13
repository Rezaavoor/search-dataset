# Document Embedding PCA

- Documents projected: 7,129
- Embedded pages processed: 171,860
- Sampled page points shown in background: 8,000
- Rows skipped due to malformed/missing embeddings: 0
- Median pages per document: 5.0
- 90th percentile pages per document: 48.0
- PCA explained variance: PC1=10.20% | PC2=5.99%

Interpretation:
- Each colored point is one document centroid computed as the mean of its L2-normalized page embeddings.
- Point size and color both increase with document page count.
- Gray points are a random sample of individual pages projected into the same PCA basis.
- PCA is a lossy 2D projection, so use it for coarse distribution structure rather than exact distance judgments.
