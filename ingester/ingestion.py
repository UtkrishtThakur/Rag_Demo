"""Local RAG ingestion pipeline.

What this script does:
- Scans a data folder for documents
- Extracts text from PDF / DOCX / TXT / MD
- Chunks text with overlap
- Generates embeddings
- Stores chunks in a local persistent vector DB (Chroma)
- Tracks per-file status in SQLite so reruns are incremental

Run:
    python ingester_main.py --data-dir ./data --db-dir ./vector_db --state-db ./vector_db/ingest_state.sqlite

Optional env vars:
    EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
    CHUNK_SIZE=900
    CHUNK_OVERLAP=150
    COLLECTION_NAME=rag_demo

Notes:
- This is intentionally local-first.
- Status values:
    pending   -> discovered but not yet processed
    processing -> currently being processed
    done      -> successfully indexed
    error     -> failed during ingestion
    skipped   -> unchanged and already indexed
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import chromadb
except Exception as exc:  # pragma: no cover
    chromadb = None
    _CHROMA_IMPORT_ERROR = exc
else:
    _CHROMA_IMPORT_ERROR = None

try:
    from docx import Document as DocxDocument
except Exception as exc:  # pragma: no cover
    DocxDocument = None
    _DOCX_IMPORT_ERROR = exc
else:
    _DOCX_IMPORT_ERROR = None

try:
    from pypdf import PdfReader
except Exception as exc:  # pragma: no cover
    PdfReader = None
    _PDF_IMPORT_ERROR = exc
else:
    _PDF_IMPORT_ERROR = None

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover
    SentenceTransformer = None
    _EMBED_IMPORT_ERROR = exc
else:
    _EMBED_IMPORT_ERROR = None


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
DEFAULT_COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_demo")
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))


@dataclass(frozen=True)
class FileRecord:
    path: Path
    sha256: str
    size_bytes: int
    modified_at: float
    ext: str


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    file_path: str
    file_name: str
    file_hash: str
    chunk_index: int
    page_number: Optional[int]
    text: str


class IngestState:
    def __init__(self, sqlite_path: Path):
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(sqlite_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS file_ingest_state (
                file_path TEXT PRIMARY KEY,
                sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                modified_at REAL NOT NULL,
                status TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                last_error TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS run_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                event TEXT NOT NULL,
                message TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def get(self, file_path: str) -> Optional[sqlite3.Row]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM file_ingest_state WHERE file_path = ?", (file_path,))
        return cur.fetchone()

    def upsert_pending(self, file: FileRecord) -> None:
        now = utc_now()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO file_ingest_state(file_path, sha256, size_bytes, modified_at, status, chunk_count, last_error, updated_at)
            VALUES (?, ?, ?, ?, 'pending', 0, NULL, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                sha256=excluded.sha256,
                size_bytes=excluded.size_bytes,
                modified_at=excluded.modified_at,
                status='pending',
                last_error=NULL,
                updated_at=excluded.updated_at
            """,
            (str(file.path), file.sha256, file.size_bytes, file.modified_at, now),
        )
        self.conn.commit()

    def mark_processing(self, file: FileRecord) -> None:
        self._set_status(file, "processing")

    def mark_done(self, file: FileRecord, chunk_count: int) -> None:
        now = utc_now()
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE file_ingest_state
            SET status='done', chunk_count=?, last_error=NULL, updated_at=?
            WHERE file_path=?
            """,
            (chunk_count, now, str(file.path)),
        )
        self.conn.commit()

    def mark_skipped(self, file: FileRecord, chunk_count: int) -> None:
        now = utc_now()
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE file_ingest_state
            SET status='skipped', chunk_count=?, last_error=NULL, updated_at=?
            WHERE file_path=?
            """,
            (chunk_count, now, str(file.path)),
        )
        self.conn.commit()

    def mark_error(self, file: FileRecord, err: Exception | str) -> None:
        now = utc_now()
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE file_ingest_state
            SET status='error', last_error=?, updated_at=?
            WHERE file_path=?
            """,
            (truncate(str(err), 2000), now, str(file.path)),
        )
        self.conn.commit()

    def log_event(self, file_path: str, sha256: str, event: str, message: str = "") -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO run_log(file_path, sha256, event, message, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (file_path, sha256, event, message, utc_now()),
        )
        self.conn.commit()

    def _set_status(self, file: FileRecord, status: str) -> None:
        now = utc_now()
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE file_ingest_state
            SET status=?, last_error=NULL, updated_at=?
            WHERE file_path=?
            """,
            (status, now, str(file.path)),
        )
        self.conn.commit()

    def counts(self) -> dict:
        cur = self.conn.cursor()
        cur.execute("SELECT status, COUNT(*) AS c FROM file_ingest_state GROUP BY status")
        return {row["status"]: row["c"] for row in cur.fetchall()}


class LocalEmbedder:
    def __init__(self, model_name: str):
        if SentenceTransformer is None:  # pragma: no cover
            raise RuntimeError(
                f"sentence-transformers is not installed: {_EMBED_IMPORT_ERROR}"
            )
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            list(texts),
            batch_size=DEFAULT_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()


class ChromaStore:
    def __init__(self, persist_dir: Path, collection_name: str):

        host = os.getenv("CHROMA_HOST", "").strip()
        port = int(os.getenv("CHROMA_PORT", "8000"))

        if host:
            logging.info(
                "Connecting to Chroma HTTP server %s:%d",
                host,
                port,
            )

            self.client = chromadb.HttpClient(
                host=host,
                port=port,
            )

        else:
            logging.info(
                "Using local Persistent Chroma at %s",
                persist_dir,
            )

            persist_dir.mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=str(persist_dir),
            )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )
    def upsert(self, chunks: List[ChunkRecord], embeddings: List[List[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = []
        for c in chunks:
            metadatas.append(
                {
                    "file_path": c.file_path,
                    "file_name": c.file_name,
                    "file_hash": c.file_hash,
                    "chunk_index": c.chunk_index,
                    "page_number": -1 if c.page_number is None else c.page_number,
                }
            )

        self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 3] + "..."


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_files(data_dir: Path) -> List[Path]:
    files: List[Path] = []
    for p in sorted(data_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(p)
    return files


def build_file_record(path: Path) -> FileRecord:
    stat = path.stat()
    return FileRecord(
        path=path.resolve(),
        sha256=file_sha256(path),
        size_bytes=stat.st_size,
        modified_at=stat.st_mtime,
        ext=path.suffix.lower(),
    )


def read_pdf(path: Path) -> List[Tuple[Optional[int], str]]:
    if PdfReader is None:  # pragma: no cover
        raise RuntimeError(f"pypdf is not installed: {_PDF_IMPORT_ERROR}")
    reader = PdfReader(str(path))
    pages: List[Tuple[Optional[int], str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def read_docx(path: Path) -> List[Tuple[Optional[int], str]]:
    if DocxDocument is None:  # pragma: no cover
        raise RuntimeError(f"python-docx is not installed: {_DOCX_IMPORT_ERROR}")
    doc = DocxDocument(str(path))
    text = "\n".join(par.text for par in doc.paragraphs if par.text and par.text.strip())
    return [(None, text)] if text.strip() else []


def read_txt_like(path: Path) -> List[Tuple[Optional[int], str]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [(None, text)] if text.strip() else []


def extract_pages(path: Path) -> List[Tuple[Optional[int], str]]:
    if path.suffix.lower() == ".pdf":
        return read_pdf(path)
    if path.suffix.lower() == ".docx":
        return read_docx(path)
    if path.suffix.lower() in {".txt", ".md"}:
        return read_txt_like(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    out = []
    prev_blank = False
    for line in lines:
        blank = not line.strip()
        if blank:
            if not prev_blank:
                out.append("")
            prev_blank = True
        else:
            out.append(line)
            prev_blank = False
    return "\n".join(out).strip()


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    def flush_current() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
            current = ""

    for para in paragraphs:
        if not current:
            if len(para) <= chunk_size:
                current = para
            else:
                # paragraph itself is large; split greedily
                start = 0
                while start < len(para):
                    end = min(len(para), start + chunk_size)
                    chunks.append(para[start:end].strip())
                    if end == len(para):
                        break
                    start = max(0, end - overlap)
            continue

        candidate = current + "\n\n" + para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            flush_current()
            if len(para) <= chunk_size:
                current = para
            else:
                start = 0
                while start < len(para):
                    end = min(len(para), start + chunk_size)
                    chunks.append(para[start:end].strip())
                    if end == len(para):
                        break
                    start = max(0, end - overlap)

    flush_current()

    # Add overlap between adjacent chunks by rolling the tail of the previous chunk
    if overlap > 0 and len(chunks) > 1:
        overlapped: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = overlapped[-1]
            tail = prev[-overlap:] if len(prev) > overlap else prev
            merged = tail + "\n\n" + chunks[i]
            if len(merged) > chunk_size:
                merged = chunks[i]
            overlapped.append(merged)
        chunks = overlapped

    return [c for c in chunks if c.strip()]


def make_chunks(file: FileRecord, pages: List[Tuple[Optional[int], str]], chunk_size: int, overlap: int) -> List[ChunkRecord]:
    out: List[ChunkRecord] = []
    chunk_index = 0
    for page_num, raw in pages:
        cleaned = clean_text(raw)
        if not cleaned:
            continue
        pieces = split_into_chunks(cleaned, chunk_size=chunk_size, overlap=overlap)
        for piece in pieces:
            out.append(
                ChunkRecord(
                    chunk_id=str(uuid.uuid4()),
                    file_path=str(file.path),
                    file_name=file.path.name,
                    file_hash=file.sha256,
                    chunk_index=chunk_index,
                    page_number=page_num,
                    text=piece,
                )
            )
            chunk_index += 1
    return out


def process_file(
    path: Path,
    state: IngestState,
    store: ChromaStore,
    embedder: LocalEmbedder,
    chunk_size: int,
    overlap: int,
) -> None:
    file = build_file_record(path)
    prev = state.get(str(file.path))

    if prev is not None and prev["sha256"] == file.sha256 and prev["status"] == "done":
        state.mark_skipped(file, int(prev["chunk_count"] or 0))
        state.log_event(str(file.path), file.sha256, "skip", "unchanged file already indexed")
        logging.info("SKIP  %s (unchanged)", file.path.name)
        return

    state.upsert_pending(file)
    state.mark_processing(file)
    state.log_event(str(file.path), file.sha256, "start", "ingestion started")
    logging.info("START %s", file.path.name)

    try:
        pages = extract_pages(path)
        if not pages:
            raise ValueError("no extractable text found")

        chunks = make_chunks(file, pages, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            raise ValueError("chunking produced zero chunks")

        embeddings: List[List[float]] = []
        texts = [c.text for c in chunks]
        for i in range(0, len(texts), DEFAULT_BATCH_SIZE):
            batch = texts[i : i + DEFAULT_BATCH_SIZE]
            embeddings.extend(embedder.encode(batch))

        store.upsert(chunks, embeddings)
        state.mark_done(file, len(chunks))
        state.log_event(str(file.path), file.sha256, "done", f"indexed {len(chunks)} chunks")
        logging.info("DONE  %s | chunks=%d", file.path.name, len(chunks))

    except Exception as exc:
        state.mark_error(file, exc)
        state.log_event(str(file.path), file.sha256, "error", str(exc))
        logging.exception("ERR   %s | %s", file.path.name, exc)


def ingest_all(data_dir: Path, db_dir: Path, state_db: Path, collection_name: str, model_name: str, chunk_size: int, overlap: int) -> None:
    if not data_dir.exists():
        raise FileNotFoundError(f"data dir does not exist: {data_dir}")

    if not data_dir.is_dir():
        raise NotADirectoryError(f"data dir is not a directory: {data_dir}")

    if _CHROMA_IMPORT_ERROR is not None:
        raise RuntimeError(f"chromadb is required: {_CHROMA_IMPORT_ERROR}")

    embedder = LocalEmbedder(model_name)
    store = ChromaStore(db_dir, collection_name)
    state = IngestState(state_db)

    files = discover_files(data_dir)
    logging.info("Found %d supported files", len(files))

    if not files:
        logging.warning("No supported documents found in %s", data_dir)
        return

    for path in files:
        process_file(path, state, store, embedder, chunk_size=chunk_size, overlap=overlap)

    counts = state.counts()
    logging.info("Status counts: %s", counts)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local RAG ingestion pipeline")
    p.add_argument("--data-dir", type=Path, default=Path("./data"), help="Folder containing source documents")
    p.add_argument("--db-dir", type=Path, default=Path("./vector_db"), help="Directory for Chroma persistence")
    p.add_argument(
        "--state-db",
        type=Path,
        default=Path("./vector_db/ingest_state.sqlite"),
        help="SQLite DB for file ingestion status",
    )
    p.add_argument("--collection-name", type=str, default=DEFAULT_COLLECTION_NAME, help="Vector collection name")
    p.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL, help="SentenceTransformer model")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size in characters")
    p.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap in characters")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        ingest_all(
            data_dir=args.data_dir,
            db_dir=args.db_dir,
            state_db=args.state_db,
            collection_name=args.collection_name,
            model_name=args.embed_model,
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
        )
        return 0
    except KeyboardInterrupt:
        logging.warning("Interrupted")
        return 130
    except Exception as exc:
        logging.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
