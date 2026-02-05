# RAGAS Synthetic Dataset Generator - Session Summary

## Overview

This session involved creating a Python script to generate synthetic Q&A datasets from legal documents using RAGAS (Retrieval-Augmented Generation Assessment). The generated datasets can be used to evaluate RAG systems.

---

## Files Created

| File | Description |
|------|-------------|
| `generate_synthetic_dataset.py` | Main script for generating synthetic datasets |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for environment variables |
| `synthetic_dataset.csv` | Generated test dataset (CSV format) |
| `synthetic_dataset.json` | Generated test dataset (JSON format) |

---

## Script Features

### Document Loading
- Supports **PDF**, **DOCX**, and **TXT** files
- Recursive directory scanning
- Individual file loading via `--files` argument
- Progress bars for loading status

### LLM Providers
- **OpenAI** (default)
- **Azure OpenAI** (auto-detected from environment variables)

### Output
- CSV and JSON formats
- Includes source file tracking for each Q&A pair
- Multiple query types: single-hop, multi-hop, abstract, specific

---

## Usage Examples

```bash
# Basic usage (5 documents, 5 samples)
python generate_synthetic_dataset.py --max-files 5 --testset-size 5

# Specific folders
python generate_synthetic_dataset.py \
  --specific-folders "Rental Agreements Archive" "Evals" \
  --testset-size 50

# Include individual files
python generate_synthetic_dataset.py \
  --specific-folders "Folder1" "Folder2" \
  --files "specific-document.pdf" \
  --testset-size 30

# Full options
python generate_synthetic_dataset.py \
  --input-dir . \
  --output synthetic_dataset \
  --testset-size 50 \
  --model gpt-4o-mini \
  --file-types pdf docx txt \
  --max-files 100 \
  --output-formats csv json \
  --provider azure
```

---

## Environment Configuration

### For OpenAI
```env
OPENAI_API_KEY=sk-your-api-key-here
```

### For Azure OpenAI
```env
AZURE_OPENAI_API_KEY=your-azure-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-large
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

---

## RAGAS Pipeline Explained

### Document → Knowledge Graph → Q&A Generation

```
Documents
    │
    ▼
┌─────────────────────────────────────────┐
│  HeadlinesExtractor [LLM]               │  Extract section headlines
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  HeadlineSplitter [ALGO]                │  Split by headlines
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  SummaryExtractor [LLM]                 │  Generate summaries
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  EmbeddingExtractor [EMBEDDINGS]        │  Create vectors
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  ThemesExtractor [LLM]                  │  Identify themes
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  NERExtractor [LLM]                     │  Extract entities
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Similarity Builders [ALGO]             │  Build relationships
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Persona Generation [LLM]               │  Create user personas
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Scenario Generation [LLM]              │  Create use cases
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Query + Answer Generation [LLM]        │  Generate Q&A pairs
└─────────────────────────────────────────┘
    │
    ▼
Final Dataset (CSV/JSON)
```

### LLM Usage by Step

| Step | Uses LLM | Uses Embeddings | Pure Algorithm |
|------|:--------:|:---------------:|:--------------:|
| HeadlinesExtractor | ✓ | | |
| HeadlineSplitter | | | ✓ |
| SummaryExtractor | ✓ | | |
| CustomNodeFilter | ✓ | | |
| EmbeddingExtractor | | ✓ | |
| ThemesExtractor | ✓ | | |
| NERExtractor | ✓ | | |
| CosineSimilarityBuilder | | | ✓ |
| OverlapScoreBuilder | | | ✓ |
| Persona Generation | ✓ | | |
| Scenario Generation | ✓ | | |
| Query Generation | ✓ | | |
| Answer Generation | ✓ | | |

---

## Query Types Generated

| Type | Description | Example |
|------|-------------|---------|
| `single_hop_specific` | Direct question from one document | "Who is the LESSOR in this agreement?" |
| `single_hop_abstract` | Conceptual question from one document | "What are the tenant's responsibilities?" |
| `multi_hop_specific` | Detailed question requiring multiple documents | "Compare rent terms between the 2014 and 2016 agreements" |
| `multi_hop_abstract` | Conceptual question spanning documents | "What are common terms across rental agreements?" |

---

## Output Schema

```json
{
  "user_input": "What are the terms of the rental agreement?",
  "reference_contexts": ["The LESSEE shall pay monthly rent of Rs. 5500..."],
  "reference": "The rental agreement specifies a monthly rent of Rs. 5500...",
  "source_files": ["54945838-Rental-Agreement.pdf.docx"],
  "source_files_readable": "54945838-Rental-Agreement.pdf.docx",
  "persona_name": "Legal Professional",
  "query_style": "formal",
  "query_length": "medium",
  "synthesizer_name": "single_hop_specific_query_synthesizer"
}
```

---

## Issues Encountered & Solutions

### 1. API Key Authentication
**Problem:** Invalid OpenAI API key format
**Solution:** OpenAI keys must start with `sk-`. Updated script to support Azure OpenAI.

### 2. Azure OpenAI Temperature Restriction
**Problem:** GPT-5/o1 models only support `temperature=1`
**Solution:** Switched to `gpt-4o-mini` deployment which supports custom temperature.

### 3. Missing Dependencies
**Problem:** `rapidfuzz` not installed
**Solution:** Added to `requirements.txt`

### 4. Large PDF Fragmentation
**Problem:** `PyPDFLoader` splits PDFs into one document per page. A 5447-page PDF creates 5447 separate documents, each lacking meaningful structure for RAGAS.
**Solution:**
- Avoid large PDFs for RAGAS
- Or merge pages before processing
- Use complete, self-contained documents (like individual contracts)

### 5. Headlines Not Found
**Problem:** Single PDF pages don't have headlines, causing `HeadlineSplitter` to fail
**Solution:** Use smaller, complete documents rather than page fragments

---

## Document Structure Requirements

RAGAS works best with **complete, self-contained documents** that have:
- Clear section headlines
- Logical structure
- Sufficient content per section

**Good for RAGAS:**
- Individual contracts (DOCX)
- Complete legal agreements
- Structured reports

**Problematic for RAGAS:**
- Large PDFs split into pages
- Fragmentary content
- Unstructured text dumps

---

## Dependencies

```
ragas>=0.3.5
python-dotenv>=1.0.0
langchain-community>=0.3.0
langchain-openai>=0.2.0
langchain>=0.3.0
pypdf>=4.0.0
docx2txt>=0.8
pandas>=2.0.0
pyarrow>=15.0.0
rapidfuzz>=3.0.0
```

---

## Test Run Results

### Successful Test (5 docs, 5 samples)
- **Input:** 5 DOCX files from Rental Agreements Archive
- **Output:** 6 Q&A pairs
- **Time:** ~2 minutes
- **Source tracking:** Working correctly

### Failed Large Run (100 docs, 50 samples)
- **Issue:** 5447-pages.pdf dominated the dataset
- **Cause:** Each PDF page became a separate document without structure
- **Fix:** Exclude large PDFs or merge pages before processing

---

## Folder Structure Analyzed

```
search-dataset/
├── 5447-pages.pdf (5447 pages - problematic for RAGAS)
├── Rental Agreements Archive/ (43 DOCX files - good)
├── Evals/ (PDFs - mixed results)
├── gdpval-documents/ (251 PDFs, 66 DOCX)
├── Claires/ (24,203 PDF pages)
├── Law_World_415-434/ (Various legal PDFs)
├── generate_synthetic_dataset.py
├── requirements.txt
├── .env.example
├── .env
├── synthetic_dataset.csv
└── synthetic_dataset.json
```

---

## Recommendations

1. **Start Small:** Test with 5-10 documents before scaling up
2. **Use DOCX:** These tend to have better structure than PDFs
3. **Avoid Mega-PDFs:** Large PDFs split into fragments don't work well
4. **Check API Costs:** Each run makes 50-100+ LLM calls
5. **Use gpt-4o-mini:** Cheaper than gpt-4o with similar quality for this task

---

## Next Steps

1. Consider merging PDF pages into complete documents before processing
2. Implement document quality filtering (skip docs that are too short)
3. Add retry logic for failed headline extractions
4. Consider alternative chunking strategies for large documents
