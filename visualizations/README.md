# Visualizations

This folder contains one-off analysis scripts and generated artifacts for
understanding the embedding corpus.

## Document PCA

`document_embedding_pca.py` reads the page-level embeddings stored in
`processed/pdf_page_store.sqlite`, aggregates them into one centroid per
document, and projects those document vectors into 2D with PCA.

It writes:

- `document_embedding_pca.html` - scatter plot of document centroids
- `document_embedding_pca_points.csv` - exported 2D coordinates per document
- `document_embedding_pca_summary.md` - quick stats and interpretation notes

The HTML plot also includes a gray background cloud of sampled page embeddings
projected into the same PCA basis, which helps show how page-level spread sits
inside the broader document-level distribution.

Run it from the project root with:

```bash
python visualizations/document_embedding_pca.py
```

Optional flags:

```bash
python visualizations/document_embedding_pca.py \
  --db processed/pdf_page_store.sqlite \
  --output-dir visualizations \
  --sample-pages 8000
```

## Document UMAP

`document_embedding_umap.py` uses the same document-centroid input as the PCA
script, but fits a 2D UMAP projection instead. This is often better for seeing
local clusters and neighborhood structure.

It writes:

- `document_embedding_umap.html` - interactive scatter plot of document centroids in UMAP space
- `document_embedding_umap_points.csv` - exported 2D coordinates per document
- `document_embedding_umap_summary.md` - corpus stats plus UMAP settings
- `vendor/umap-js.min.js` - local browser bundle used by the interactive HTML page

The generated UMAP HTML is interactive:

- `n_neighbors` can be adjusted live in the page
- `min_dist` can be adjusted live in the page
- the plot reruns UMAP in the browser using PCA-reduced document-centroid features
- the page uses a local JS bundle, so it does not depend on loading UMAP from a CDN at view time

Run it from the project root with:

```bash
python visualizations/document_embedding_umap.py
```

Optional flags:

```bash
python visualizations/document_embedding_umap.py \
  --db processed/pdf_page_store.sqlite \
  --output-dir visualizations \
  --sample-pages 8000 \
  --n-neighbors 25 \
  --min-dist 0.08 \
  --metric cosine
```

Notes:

- The Python script requires the optional `umap-learn` package:

```bash
python -m pip install umap-learn
```

- The interactive HTML focuses on document centroids for responsiveness. The
  sampled page cloud is still included in the static PCA visualization, but not
  in the live browser-side UMAP reruns.
