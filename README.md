# Local AI Document Processing Pipeline

A fully local AI system for document ingestion, classification, structured field extraction, semantic search, and question answering. No paid or hosted APIs are used — everything runs on your machine.

---

## What It Does

| Step | Description |
|------|-------------|
| **Ingest** | Reads all PDF and TXT files from a folder |
| **Classify** | Labels each document as Invoice, Resume, Utility Bill, Other, or Unclassifiable |
| **Extract** | Pulls structured fields based on document type (see table below) |
| **Search** | Semantic search over all documents using local embeddings + FAISS |
| **QA (Bonus)** | Natural language question answering using a local LLM (TinyLlama) |
| **Output** | Saves `Output.json` with all results |

### Extracted Fields per Document Type

| Type | Fields |
|------|--------|
| Invoice | `invoice_number`, `date`, `company`, `total_amount` |
| Resume | `name`, `email`, `phone`, `experience_years` |
| Utility Bill | `account_number`, `date`, `usage_kwh`, `amount_due` |
| Other / Unclassifiable | *(no extraction required)* |

---

## Installation

### 1. Prerequisites
- Python 3.8 or higher

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** 
> On first run, the following models are downloaded automatically and cached locally.
> After that, the system runs fully offline — no internet required.
>
> | Model | Purpose | Cache Location |
> |-------|---------|----------------|
> | `facebook/bart-large-mnli` | Zero-shot document classification | `~/.cache/huggingface/hub/` |
> | `all-MiniLM-L6-v2` | Sentence embeddings for semantic search | `~/.cache/torch/sentence_transformers/` |
> | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Local LLM for question answering | `~/.cache/huggingface/hub/` |

---

## How to Run

### — Streamlit UI

```bash
streamlit run app.py / python -m streamlit run app.py
```

Then open `http://localhost:8501` in your browser. Upload PDFs via the sidebar and click **Process Documents**. Once processed, type any question in the search box to get a direct answer from the local LLM.

---

## Output Format

`Output.json`
---

## Libraries & Methods

| Library | Version | Purpose |
|---------|---------|---------|
| `pypdf` | ≥4.0.0 | PDF text extraction and page parsing |
| `transformers` | ≥4.38.0 | BART zero-shot classifier + TinyLlama LLM loading |
| `accelerate` | ≥0.26.0 | Required for loading transformer models on CPU |
| `sentence-transformers` | ≥2.5.0 | Local sentence embeddings (`all-MiniLM-L6-v2`) |
| `faiss-cpu` | ≥1.7.4 | Vector similarity index (cosine via inner product on L2-normalized vectors) |
| `numpy` | ≥1.24.0 | Embedding normalization and vector operations |
| `torch` | ≥2.0.0 | Backend runtime for all transformer models |
| `streamlit` | ≥1.32.0 | Optional web UI |

### Classification Approach
A two-stage strategy is used:
1. **Keyword scoring** — fast heuristic pass using structural signals (e.g. "Bill To", "Total Due", "kWh") with a confidence threshold
2. **BART zero-shot classifier** (`facebook/bart-large-mnli`) — used for ambiguous documents where keyword scores are too close to call

This avoids hardcoding per-company logic and generalizes to unseen document formats.

### Semantic Search Approach
- Documents are split into paragraphs and encoded using `all-MiniLM-L6-v2`
- Embeddings are L2-normalized and indexed in a FAISS `IndexFlatIP` (inner product = cosine similarity)
- At query time, results are filtered by a cosine similarity threshold (≥0.25) to return only genuinely relevant documents.

### Question Answering (Optional Bonus)
- Uses `TinyLlama-1.1B-Chat-v1.0` — a fully local, open-source LLM
- Retrieved document chunks are passed as context to the LLM
- The LLM generates a grounded natural language answer based only on the provided context
- Runs entirely on CPU — no GPU or internet required
- Expected response time: 15–45 seconds on CPU depending on hardware

---

## Project Structure

```
├── pipeline.py        # Core AI logic (ingest, classify, extract, search, QA)
├── app.py             # Streamlit UI 
├── Output.json        # Auto-generated results after processing
├── README.md          # This file
└── requirements.txt   # Python dependencies
```

---
