from uuid import uuid4

from .errors import InvalidQuestionError
from .run import RAGRun
from .schemas import utc_now_iso


class RAGInspector:
    """Entry point for explicitly recording RAG pipeline executions."""

    def start_run(
        self,
        question: str,
        *,
        query_id: str | None = None,
        rag_version: str | None = None,
        metadata: dict | None = None,
    ) -> RAGRun:
        if not isinstance(question, str) or not question.strip():
            raise InvalidQuestionError(
                "Required pipeline input 'question' must be a non-empty string."
            )
        if query_id is not None and (not isinstance(query_id, str) or not query_id.strip()):
            raise ValueError("'query_id' must be a non-empty string when provided.")
        if rag_version is not None and (not isinstance(rag_version, str) or not rag_version.strip()):
            raise ValueError("'rag_version' must be a non-empty string when provided.")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("'metadata' must be a dictionary when provided.")

        return RAGRun(
            run_id=str(uuid4()),
            question=question.strip(),
            started_at=utc_now_iso(),
            query_id=query_id.strip() if query_id else None,
            rag_version=rag_version.strip() if rag_version else None,
            metadata=dict(metadata or {}),
        )
