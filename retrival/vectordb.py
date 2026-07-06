"""Chroma vector database wrapper for retrieval.

This module is designed to be used by retrieval.py.

It supports three modes:

1) auto (default)
   - If CHROMA_HOST is set, it prefers HTTP mode.
   - If the HTTP collection is empty but the persistent collection has data,
     it falls back to the persistent store so a demo does not break when
     ingestion wrote to disk instead of the running Chroma server.

2) http
   - Always use the Chroma HTTP service.

3) persistent
   - Always use the local persistent database path.

Environment variables:
   CHROMA_MODE=auto|http|persistent
   CHROMA_HOST=chroma
   CHROMA_PORT=8000
   CHROMA_SSL=false
   CHROMA_PATH=./vector_db
   COLLECTION_NAME=rag_demo
   TOP_K=5

Example:
    from vectordb import ChromaVectorDB

    db = ChromaVectorDB()
    results = db.search(query_embedding, top_k=5)

Returned search results are normalized into plain Python objects so the
retrieval layer does not depend on Chroma-specific response shapes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import chromadb
except Exception as exc:  # pragma: no cover
    chromadb = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

try:
    from config import (
        CHROMA_HOST,
        CHROMA_PATH,
        CHROMA_PORT,
        CHROMA_SSL,
        COLLECTION_NAME,
        TOP_K,
    )
except Exception:  # pragma: no cover
    # Fall back to env vars if config.py is not importable for any reason.
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_demo")
    CHROMA_HOST = os.getenv("CHROMA_HOST", "").strip()
    CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
    CHROMA_SSL = os.getenv("CHROMA_SSL", "false").lower() in {"1", "true", "yes", "on"}
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./vector_db").strip()
    TOP_K = int(os.getenv("TOP_K", "5"))


DEFAULT_CHROMA_MODE = os.getenv("CHROMA_MODE", "auto").strip().lower()
DEFAULT_COLLECTION_NAME = COLLECTION_NAME
DEFAULT_CHROMA_HOST = CHROMA_HOST
DEFAULT_CHROMA_PORT = CHROMA_PORT
DEFAULT_CHROMA_SSL = CHROMA_SSL
DEFAULT_CHROMA_PATH = CHROMA_PATH
DEFAULT_TOP_K = TOP_K


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    distance: Optional[float]
    metadata: Dict[str, Any]


class ChromaVectorDB:
    """Small, defensive wrapper around Chroma.

    The important bit here is that retrieval can survive the common demo-case
    mismatch where ingestion wrote to the persistent database, but retrieval
    is pointed at the HTTP service (or vice versa).
    """

    def __init__(
        self,
        collection_name: str | None = None,
        chroma_mode: str | None = None,
        chroma_host: str | None = None,
        chroma_port: int | None = None,
        chroma_ssl: bool | None = None,
        chroma_path: str | None = None,
    ) -> None:
        if chromadb is None:  # pragma: no cover
            raise RuntimeError(f"chromadb is required but could not be imported: {_IMPORT_ERROR}")

        self.collection_name = collection_name or DEFAULT_COLLECTION_NAME
        self.chroma_mode = (chroma_mode or DEFAULT_CHROMA_MODE or "auto").strip().lower()
        self.chroma_host = (chroma_host if chroma_host is not None else DEFAULT_CHROMA_HOST).strip()
        self.chroma_port = chroma_port if chroma_port is not None else DEFAULT_CHROMA_PORT
        self.chroma_ssl = chroma_ssl if chroma_ssl is not None else DEFAULT_CHROMA_SSL
        self.chroma_path = (chroma_path if chroma_path is not None else DEFAULT_CHROMA_PATH).strip()

        self._http_client = None
        self._persistent_client = None
        self._active_client = None
        self._active_collection = None
        self._active_mode = None

        self._initialize_clients()
        self._select_active_collection()

    def _initialize_clients(self) -> None:
        """Create the configured Chroma clients up front."""
        if self.chroma_mode not in {"auto", "http", "persistent"}:
            raise ValueError("CHROMA_MODE must be one of: auto, http, persistent")

        if self.chroma_mode in {"auto", "http"} and self.chroma_host:
            self._http_client = chromadb.HttpClient(
                host=self.chroma_host,
                port=self.chroma_port,
                ssl=self.chroma_ssl,
            )

        if self.chroma_mode in {"auto", "persistent"}:
            self._persistent_client = chromadb.PersistentClient(path=self.chroma_path)

        # If CHROMA_MODE=http but host is not set, fail fast.
        if self.chroma_mode == "http" and not self._http_client:
            raise ValueError("CHROMA_MODE=http requires CHROMA_HOST to be set")

        # If CHROMA_MODE=persistent, make sure we actually created it.
        if self.chroma_mode == "persistent" and not self._persistent_client:
            self._persistent_client = chromadb.PersistentClient(path=self.chroma_path)

    def _get_collection_from_client(self, client):
        try:
            return client.get_collection(name=self.collection_name)
        except Exception:
            return client.get_or_create_collection(name=self.collection_name)

    def _collection_count(self, collection) -> int:
        try:
            return int(collection.count())
        except Exception:
            return 0

    def _select_active_collection(self) -> None:
        """Pick the best collection for retrieval.

        Priority:
        - explicit http -> HTTP
        - explicit persistent -> persistent
        - auto -> prefer HTTP when it has data, otherwise fall back to persistent
        """
        http_collection = None
        persistent_collection = None
        http_count = None
        persistent_count = None

        if self._http_client is not None:
            http_collection = self._get_collection_from_client(self._http_client)
            http_count = self._collection_count(http_collection)

        if self._persistent_client is not None:
            persistent_collection = self._get_collection_from_client(self._persistent_client)
            persistent_count = self._collection_count(persistent_collection)

        if self.chroma_mode == "http":
            self._active_client = self._http_client
            self._active_collection = http_collection
            self._active_mode = "http"
            return

        if self.chroma_mode == "persistent":
            self._active_client = self._persistent_client
            self._active_collection = persistent_collection
            self._active_mode = "persistent"
            return

        # auto mode
        if http_collection is not None and (http_count or 0) > 0:
            self._active_client = self._http_client
            self._active_collection = http_collection
            self._active_mode = "http"
            return

        if persistent_collection is not None and (persistent_count or 0) > 0:
            self._active_client = self._persistent_client
            self._active_collection = persistent_collection
            self._active_mode = "persistent"
            return

        # If neither has data, still prefer HTTP when configured so new writes
        # during debugging go to the running service.
        if http_collection is not None:
            self._active_client = self._http_client
            self._active_collection = http_collection
            self._active_mode = "http"
            return

        if persistent_collection is not None:
            self._active_client = self._persistent_client
            self._active_collection = persistent_collection
            self._active_mode = "persistent"
            return

        raise RuntimeError(
            "Could not initialize any Chroma collection. Check CHROMA_HOST/CHROMA_PATH and service availability."
        )

    @property
    def active_mode(self) -> str:
        return self._active_mode or "unknown"

    def count(self) -> int:
        return int(self._active_collection.count())

    def peek(self, limit: int = 5) -> Dict[str, Any]:
        if limit <= 0:
            raise ValueError("limit must be > 0")
        return self._active_collection.peek(limit)

    def get_by_ids(self, ids: Sequence[str]) -> Dict[str, Any]:
        ids = list(ids)
        if not ids:
            raise ValueError("ids must not be empty")
        return self._active_collection.get(ids=ids, include=["documents", "metadatas", "embeddings"])

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = DEFAULT_TOP_K,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Search the active collection and return normalized results."""
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        embedding = list(query_embedding)
        if not embedding:
            raise ValueError("query_embedding must not be empty")

        result = self._active_collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        chunks: List[RetrievedChunk] = []
        for idx, doc in enumerate(docs):
            chunk_id = ids[idx] if idx < len(ids) else ""
            meta = metas[idx] if idx < len(metas) and metas[idx] is not None else {}
            dist = dists[idx] if idx < len(dists) else None
            chunks.append(
                RetrievedChunk(
                    id=chunk_id,
                    text=doc or "",
                    distance=float(dist) if dist is not None else None,
                    metadata=dict(meta),
                )
            )
        return chunks

    def search_text(
        self,
        query_embedding: Sequence[float],
        top_k: int = DEFAULT_TOP_K,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "id": chunk.id,
                "text": chunk.text,
                "distance": chunk.distance,
                "metadata": chunk.metadata,
            }
            for chunk in self.search(query_embedding=query_embedding, top_k=top_k, where=where)
        ]

    def health_check(self) -> Dict[str, Any]:
        """Return a detailed status payload for debugging."""
        try:
            data = {
                "ok": True,
                "collection_name": self.collection_name,
                "active_mode": self.active_mode,
                "active_count": self.count(),
                "http_count": None,
                "persistent_count": None,
                "http_available": self._http_client is not None,
                "persistent_available": self._persistent_client is not None,
                "host": self.chroma_host or None,
                "port": self.chroma_port if self.chroma_host else None,
                "path": self.chroma_path if self._persistent_client is not None else None,
            }
            if self._http_client is not None:
                data["http_count"] = self._collection_count(self._get_collection_from_client(self._http_client))
            if self._persistent_client is not None:
                data["persistent_count"] = self._collection_count(self._get_collection_from_client(self._persistent_client))
            return data
        except Exception as exc:
            return {
                "ok": False,
                "collection_name": self.collection_name,
                "error": str(exc),
            }

    def debug_state(self) -> Dict[str, Any]:
        return self.health_check()


if __name__ == "__main__":
    db = ChromaVectorDB()
    print(db.health_check())
    print("count=", db.count())
    print("peek_keys=", list(db.peek(1).keys()))

