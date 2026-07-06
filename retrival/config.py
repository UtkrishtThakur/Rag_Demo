"""Central configuration for the local RAG demo.

This file keeps all runtime settings in one place so the rest of the codebase
can import from here instead of scattering environment-variable lookups.

Defaults are tuned for your current setup:
- Local Chroma in Docker
- Ollama on the host machine
- qwen3:8b as the answer model
- sentence-transformers/all-MiniLM-L6-v2 for embeddings
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
VECTOR_DB_DIR = Path(os.getenv("VECTOR_DB_DIR", str(PROJECT_ROOT / "vector_db")))
STATE_DB_PATH = Path(os.getenv("STATE_DB_PATH", str(VECTOR_DB_DIR / "ingest_state.sqlite")))
PROMPT_FILE = Path(os.getenv("PROMPT_FILE", str(PROJECT_ROOT / "prompt.txt")))


COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_demo")

# Embeddings
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))
RETRIEVAL_LOG_LEVEL = os.getenv("RETRIEVAL_LOG_LEVEL", "INFO")

# Chroma
CHROMA_HOST = os.getenv("CHROMA_HOST", "")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_SSL = os.getenv("CHROMA_SSL", "false").lower() in {"1", "true", "yes", "on"}
CHROMA_PATH = os.getenv("CHROMA_PATH", str(VECTOR_DB_DIR))

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
OLLAMA_TOP_P = float(os.getenv("OLLAMA_TOP_P", "0.9"))
OLLAMA_TOP_K = int(os.getenv("OLLAMA_TOP_K", "40"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))
OLLAMA_REPEAT_PENALTY = float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.1"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
OLLAMA_RETRIES = int(os.getenv("OLLAMA_RETRIES", "1"))
OLLAMA_STREAM = os.getenv("OLLAMA_STREAM", "false").lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppConfig:
    data_dir: Path = DATA_DIR
    vector_db_dir: Path = VECTOR_DB_DIR
    state_db_path: Path = STATE_DB_PATH
    prompt_file: Path = PROMPT_FILE
    collection_name: str = COLLECTION_NAME

    embed_model: str = EMBED_MODEL
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    embed_batch_size: int = EMBED_BATCH_SIZE

    top_k: int = TOP_K
    retrieval_log_level: str = RETRIEVAL_LOG_LEVEL

    chroma_host: str = CHROMA_HOST
    chroma_port: int = CHROMA_PORT
    chroma_ssl: bool = CHROMA_SSL
    chroma_path: str = CHROMA_PATH

    ollama_host: str = OLLAMA_HOST
    ollama_model: str = OLLAMA_MODEL
    ollama_temperature: float = OLLAMA_TEMPERATURE
    ollama_top_p: float = OLLAMA_TOP_P
    ollama_top_k: int = OLLAMA_TOP_K
    ollama_num_ctx: int = OLLAMA_NUM_CTX
    ollama_num_predict: int = OLLAMA_NUM_PREDICT
    ollama_repeat_penalty: float = OLLAMA_REPEAT_PENALTY
    ollama_timeout: int = OLLAMA_TIMEOUT
    ollama_retries: int = OLLAMA_RETRIES
    ollama_stream: bool = OLLAMA_STREAM


CONFIG = AppConfig()


def ensure_directories() -> None:
    """Create required local directories if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def as_dict() -> dict:
    """Return a plain dictionary for debugging or logging."""
    return {
        "data_dir": str(DATA_DIR),
        "vector_db_dir": str(VECTOR_DB_DIR),
        "state_db_path": str(STATE_DB_PATH),
        "prompt_file": str(PROMPT_FILE),
        "collection_name": COLLECTION_NAME,
        "embed_model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embed_batch_size": EMBED_BATCH_SIZE,
        "top_k": TOP_K,
        "retrieval_log_level": RETRIEVAL_LOG_LEVEL,
        "chroma_host": CHROMA_HOST,
        "chroma_port": CHROMA_PORT,
        "chroma_ssl": CHROMA_SSL,
        "chroma_path": CHROMA_PATH,
        "ollama_host": OLLAMA_HOST,
        "ollama_model": OLLAMA_MODEL,
        "ollama_temperature": OLLAMA_TEMPERATURE,
        "ollama_top_p": OLLAMA_TOP_P,
        "ollama_top_k": OLLAMA_TOP_K,
        "ollama_num_ctx": OLLAMA_NUM_CTX,
        "ollama_num_predict": OLLAMA_NUM_PREDICT,
        "ollama_repeat_penalty": OLLAMA_REPEAT_PENALTY,
        "ollama_timeout": OLLAMA_TIMEOUT,
        "ollama_retries": OLLAMA_RETRIES,
        "ollama_stream": OLLAMA_STREAM,
    }


if __name__ == "__main__":
    ensure_directories()
    from pprint import pprint

    pprint(as_dict())
