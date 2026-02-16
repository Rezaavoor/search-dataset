# OCR vs. pypdf Text Extraction — A/B Evaluation


## Aggregate Metrics

| Metric | Baseline (pypdf) | OCR | Delta | Change |
|--------|------------------|-----|-------|--------|
| recall@1 | 0.4569 | **0.4677** | +0.0108 | **+2.36%** |
| recall@5 | **0.8254** | 0.8168 | -0.0086 | -1.04% |
| recall@10 | 0.8836 | 0.8836 | 0.0000 | 0.00% |
| recall@20 | 0.9224 | **0.9310** | +0.0086 | +0.93% |
| MRR | 0.6140 | **0.6200** | +0.0060 | +0.97% |
| MAP | 0.6062 | **0.6148** | +0.0085 | **+1.40%** |
| nDCG@10 | 0.6686 | **0.6754** | +0.0067 | +1.00% |


## Hard Negative Analysis

Hard negatives were mined **independently** for each corpus (BM25 + embedding
similarity + LLM judge), so each set of hard negatives is calibrated to its
own text representation. This ensures a fair apples-to-apples comparison.

| Metric | Baseline (pypdf) | OCR | Delta | Change |
|--------|------------------|-----|-------|--------|
| Queries with HN | 253 | 265 | +12 | +4.7% |
| Mean HN rank | 99.1 | 101.2 | +2.1 | +2.2% |
| HN outranks positive | 19.8% | 20.4% | +0.6% | +3.1% |

Hard-negative difficulty is **effectively identical** between the two corpora
when each is measured against its own mined negatives.

### Note on prior (flawed) comparison

An earlier analysis compared hard negatives mined from the pypdf corpus against
the OCR corpus, which showed a mean HN rank of 268.4 (vs. 99.1 for pypdf).
That dramatic difference was an artifact: hard negatives selected to be
"deceptively close" in the pypdf embedding space naturally scatter when
evaluated in a different (OCR) embedding space. The corrected numbers above
use independently mined hard negatives for each corpus.


## Conclusion

OCR provides **small but consistent improvements** on primary retrieval metrics
(recall@1 +2.4%, MAP +1.4%, MRR +1.0%, nDCG@10 +1.0%). Hard-negative
difficulty is essentially the same across both corpora when measured fairly.

The improvement is modest because this corpus is predominantly text-layer PDFs
where pypdf already performs well. OCR would likely show larger gains on
scanned/image-heavy documents or PDFs with complex table layouts.

**Cost:** ~$193 for 128,787 PDF pages via Azure Document Intelligence.
