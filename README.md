# Industry-Level Multimodal RAG

A Streamlit application for ingesting enterprise documents and answering questions with a multimodal retrieval-augmented generation (RAG) workflow. It extracts text, tables, and images from office documents, stores searchable chunks in PostgreSQL with PGVector, then uses hybrid retrieval and reranking to produce grounded answers.

## What it does

- Ingests **PDF, DOCX, PPTX, XLSX, and CSV** files.
- Uses Unstructured's high-resolution partitioning and title-aware chunking.
- Preserves raw text, HTML tables, and base64 image payloads with every chunk.
- Creates LLM-enhanced descriptions for chunks containing tables or images.
- Stores embeddings from `BAAI/bge-small-en-v1.5` in PostgreSQL/PGVector.
- Expands each question into three variations, combines MMR vector search with BM25, applies reciprocal-rank fusion, and reranks the result with `BAAI/bge-reranker-base`.
- Uses Groq as the primary LLM, with OpenRouter and a local vLLM endpoint configured as fallbacks.

## Retrieval flow

```mermaid
flowchart LR
    A[Question] --> B[Query expansion]
    B --> C[MMR vector search]
    B --> D[BM25 search]
    C --> E[Reciprocal-rank fusion]
    D --> E
    E --> F[Cross-encoder reranking]
    F --> G[Grounded answer]
    H[(PostgreSQL + PGVector)] --> C
```

## Project layout

```text
src/
  Config.py                 # Models and LLM fallback configuration
  ingestion_pipeline.py     # Partition, chunk, enrich, and index documents
  retrival_methods.py       # Hybrid retrieval, RRF, reranking, and synthesis
  main.py                   # Streamlit user interface
datasets/datasets/          # Sample corporate documents
test/                       # Unit and opt-in integration tests
packages.txt                # Ubuntu system packages for document parsing
```

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- PostgreSQL with the `pgvector` extension enabled
- A Groq API key and OpenRouter API key
- A local vLLM server at `http://localhost:8005/v1` if the final fallback is to be usable

For PDF/OCR and office-document parsing, install the system dependencies. On Debian/Ubuntu:

```bash
sudo apt-get update
sudo xargs -a packages.txt apt-get install -y
```

`packages.txt` includes LibreOffice, Pandoc, Tesseract, Poppler, and file-type libraries. macOS users can install the comparable tools with Homebrew (`libreoffice`, `pandoc`, `tesseract`, `poppler`, and `libmagic`).

## Setup

Install the locked Python environment:

```bash
uv sync
```

Create a `.env` file in the repository root:

```dotenv
GROQ_API_KEY=your_groq_key
OPENROUTER_API_KEY=your_openrouter_key
DATABASE_URL=postgresql+psycopg://username:password@host:5432/database

# Optional: enables LangSmith tracing.
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true
```

The connection string must point to a database where the `vector` extension is available. For a local PostgreSQL instance, enable it once:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

> The app builds all three LLM clients during startup, so `GROQ_API_KEY` and `OPENROUTER_API_KEY` must both be set. The vLLM URL and placeholder API key are currently configured in `src/Config.py`.

## Run the app

Start the Streamlit UI from the repository root:

```bash
uv run streamlit run src/main.py
```

The UI has two modes:

- **Easy Build** queries the `DemoRAG` PGVector collection. Ingest documents into that collection before using this mode.
- **Upload Document** saves the selected file to `uploads/`, creates a UUID-named collection for it, and lets you query it in the same session.

## Ingest and query from Python

The pipeline functions can be called directly. They use Streamlit status widgets, so the Streamlit app is the recommended interface.

```python
from src.ingestion_pipeline import ingestion_pipeline
from src.retrival_methods import main_retrival_pipeline

# Index a file or directory into the default collection.
documents, elapsed = ingestion_pipeline(
    "datasets/datasets",
    collection="DemoRAG",
)

answer, elapsed = main_retrival_pipeline(
    "Who are the executives?",
    collection="DemoRAG",
)
print(answer)
```

To run either module manually:

```bash
uv run python -m src.ingestion_pipeline
uv run python -m src.retrival_methods
```

The module defaults are `test_documents` for ingestion and `DemoRAG` for retrieval; direct function calls are preferable when using another input path or collection.

## Testing

Run the mocked unit tests:

```bash
uv run pytest -m "not integration"
```

Run integration tests only when a reachable database and live credentials are intentionally available:

```bash
INTEGRATION=true uv run pytest -m integration
```

## Operational notes

- The first run downloads the embedding and reranker models from Hugging Face.
- Ingestion can be slow for image-heavy documents because multimodal chunks are summarized through the LLM.
- Collections are isolated by name. Keep the collection used at ingestion and retrieval the same.
- Uploaded files remain in `uploads/`; remove them according to your data-retention requirements.
- The repository includes synthetic sample documents in `datasets/datasets`. Treat any output as document-grounded assistance and verify important information against the source material.

## Current stack

| Concern | Implementation |
| --- | --- |
| UI | Streamlit |
| Parsing and chunking | Unstructured |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Vector database | PostgreSQL + PGVector |
| Sparse retrieval | BM25 |
| Fusion | Reciprocal-rank fusion |
| Reranking | `BAAI/bge-reranker-base` |
| LLM routing | Groq → OpenRouter → local vLLM |
| Observability | LangSmith tracing (optional) |
