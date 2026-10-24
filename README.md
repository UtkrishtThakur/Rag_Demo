# RagDemo: Local RAG Pipeline & RAG Inspector SDK

RagDemo is a local, privacy-first Retrieval-Augmented Generation (RAG) platform incorporating the **RAG Inspector SDK** (`rag-inspector-sdk`), a framework-independent Python SDK for execution tracing, information retrieval evaluation, failure diagnosis, version comparison, and benchmarking.

---

## Architecture Overview

```
RagDemo Platform
├── ingester/               # Document scanner, chunker, embedder, and Chroma vector DB loader
├── retrival/               # Query embedding, vector retrieval, Ollama generation, and SDK evaluation CLI
├── frontend/               # Streamlit web UI (document ingestion, chat, source inspection)
├── data/                   # Input documents (PDF, DOCX, TXT, MD)
├── vector_db/              # Persistent Chroma vector store and SQLite ingest state
└── rag-inspector-sdk/      # Independent RAG Inspector Python SDK
```

---

## Prerequisites

- **Docker & Docker Compose** (Recommended runtime environment)
- **Ollama** running locally on the host or in an accessible network endpoint (default: `http://127.0.0.1:11434`, model: `qwen3:8b`)

---

## Docker Development Workflow

### 1. Build and Start Services
```bash
docker compose build app
docker compose up -d
```

### 2. Enter Interactive Bash Shell
To enter the running application container and inspect the pipeline interactively:
```bash
docker compose exec app bash
```

Inside the container, verify the environment and SDK:
```bash
python --version
pip show rag-inspector-sdk
python -c "import rag_inspector; print(rag_inspector.__file__)"
```

### 3. Run Automated SDK & Integration Tests
Inside the running container or from the host:
```bash
docker compose exec app pytest rag-inspector-sdk/tests -v
```

---

## Ingestion & Retrieval CLI Workflow

### Document Ingestion
To index files from `./data` into the Chroma vector database:
```bash
docker compose exec app python ingester/ingestion.py --data-dir ./data --db-dir ./vector_db --state-db ./vector_db/ingest_state.sqlite
```

### Interactive Retrieval & Tracing CLI
To run queries interactively with Chroma retrieval, Ollama generation, and automated SDK execution tracing:
```bash
docker compose exec app python retrival/retrieval.py
```

Or ask a single question non-interactively:
```bash
docker compose exec app python retrival/retrieval.py --question "What is the curriculum for Artificial Intelligence?"
```

---

## Offline Evaluation & Failure Diagnostics

RagDemo provides an offline evaluation script (`retrival/evaluate.py`) that matches recorded execution runs against ground-truth datasets and performs diagnostic classification:

```bash
docker compose exec app python retrival/evaluate.py \
    --dataset path/to/dataset.json \
    --executions path/to/executions.json \
    --output-dir ./evaluation_output
```

This outputs both structured JSON (`evaluation_report.json`) and Markdown summaries (`evaluation_report.md`).

---

## Environment Variables

Copy `.env.example` to `.env` to customize settings:

| Variable | Default Value | Description |
|---|---|---|
| `CHROMA_HOST` | `127.0.0.1` | Chroma vector database host |
| `CHROMA_PORT` | `8000` | Chroma HTTP service port |
| `CHROMA_MODE` | `http` | Chroma connection mode (`http`, `persistent`, or `auto`) |
| `COLLECTION_NAME` | `rag_demo` | Target Chroma collection name |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model identifier |
| `CHUNK_SIZE` | `900` | Text chunk character length |
| `CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks |
| `TOP_K` | `5` | Number of chunks to retrieve per query |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama service endpoint |
| `OLLAMA_MODEL` | `qwen3:8b` | Ollama generation model |
| `RETRIEVAL_LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## Security & Archive Hygiene

The following files and directories are excluded from git and docker layers:
- `.env` (contains local deployment variables; use `.env.example` for templates)
- `vector_db/` and SQLite databases (persisted application state)
- `data/` (local user documents)
- `__pycache__/`, `*.pyc`, `.pytest_cache/`
