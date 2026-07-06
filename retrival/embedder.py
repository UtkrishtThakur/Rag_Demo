"""Query embedding utility for retrieval.

This module must use the exact same embedding model as ingestion.
It is intentionally small and self-contained so retrieval.py can call it directly.

Usage:
    from embedder import QueryEmbedder

    embedder = QueryEmbedder()
    vector = embedder.embed_query("What is the payment due date?")
    vectors = embedder.embed_queries(["q1", "q2"])
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Sequence

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover
    SentenceTransformer = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


DEFAULT_EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)


class QueryEmbedder:
    """Small wrapper around SentenceTransformer for query embeddings."""

    def __init__(self, model_name: str | None = None):
        if SentenceTransformer is None:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is required but could not be imported: "
                f"{_IMPORT_ERROR}"
            )

        self.model_name = model_name or DEFAULT_EMBED_MODEL
        self.model = self._load_model(self.model_name)

    @staticmethod
    @lru_cache(maxsize=4)
    def _load_model(model_name: str) -> SentenceTransformer:
        # Cache model instances so repeated imports / instantiations stay fast.
        return SentenceTransformer(model_name)

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query into a normalized vector."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        vector = self.model.encode(
            query.strip(),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()

    def embed_queries(self, queries: Sequence[str]) -> List[List[float]]:
        """Embed multiple queries into normalized vectors."""
        cleaned: List[str] = []
        for q in queries:
            if not isinstance(q, str) or not q.strip():
                raise ValueError("all queries must be non-empty strings")
            cleaned.append(q.strip())

        vectors = self.model.encode(
            cleaned,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())


if __name__ == "__main__":
    # Small smoke test.
    embedder = QueryEmbedder()
    vec = embedder.embed_query("test query")
    print(f"model={embedder.model_name}")
    print(f"dimension={embedder.dimension}")
    print(f"vector_len={len(vec)}")
