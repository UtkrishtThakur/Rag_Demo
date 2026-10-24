from collections.abc import Sequence
from typing import Any

from .errors import (
    InvalidAnswerError,
    InvalidDocumentError,
    InvalidRetrievalError,
    MissingPipelineDataError,
    RunStateError,
)
from .schemas import DocumentRecord, ExecutionRecord, RunStatus, utc_now_iso


class RAGRun:
    """Capture the required data from one RAG pipeline execution."""

    def __init__(
        self,
        *,
        run_id: str,
        question: str,
        started_at: str,
        query_id: str | None = None,
        rag_version: str | None = None,
        metadata: dict | None = None,
    ):
        self._record = ExecutionRecord(
            run_id=run_id,
            question=question,
            started_at=started_at,
            query_id=query_id,
            rag_version=rag_version,
            metadata=dict(metadata or {}),
        )

    @property
    def run_id(self) -> str:
        return self._record.run_id

    @property
    def status(self) -> RunStatus:
        return self._record.status

    def record_retrieval(
        self,
        documents: Sequence[str | dict[str, Any]],
        *,
        latency_ms: float | None = None,
    ) -> "RAGRun":
        self._ensure_recording()
        if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
            raise InvalidRetrievalError(
                "'documents' must be a sequence of strings or dictionaries; "
                "pass [] to explicitly record zero retrieved documents."
            )

        normalised: list[DocumentRecord] = []
        for index, document in enumerate(documents):
            score = None
            rank = None
            if isinstance(document, str):
                text = document
                doc_id = None
                metadata = {}
            elif isinstance(document, dict):
                text = document.get("text")
                doc_id = document.get("id")
                metadata = document.get("metadata", {})
                score = document.get("score")
                rank = document.get("rank")
                if not isinstance(text, str) or not text.strip():
                    raise InvalidDocumentError(
                        f"Retrieved document at index {index} requires non-empty string field 'text'."
                    )
                if doc_id is not None and not isinstance(doc_id, str):
                    raise InvalidDocumentError(
                        f"Retrieved document at index {index}: 'id' must be a string when provided."
                    )
                if not isinstance(metadata, dict):
                    raise InvalidDocumentError(
                        f"Retrieved document at index {index}: 'metadata' must be a dictionary."
                    )
            else:
                raise InvalidDocumentError(
                    f"Retrieved document at index {index} must be a string or dictionary; "
                    f"received {type(document).__name__}. Convert framework objects explicitly."
                )

            if not isinstance(text, str) or not text.strip():
                raise InvalidDocumentError(
                    f"Retrieved document at index {index} must contain non-empty text."
                )
            normalised.append(
                DocumentRecord(
                    text=text,
                    id=doc_id,
                    metadata=dict(metadata),
                    score=float(score) if score is not None else None,
                    rank=int(rank) if rank is not None else (index + 1),
                )
            )

        self._record.retrieved_documents = normalised
        if latency_ms is not None:
            self._record.retrieval_latency_ms = float(latency_ms)
        return self

    def record_context(self, supplied_context: str) -> "RAGRun":
        self._ensure_recording()
        if not isinstance(supplied_context, str):
            raise ValueError("'supplied_context' must be a string.")
        self._record.supplied_context = supplied_context
        return self

    def record_generation(
        self,
        answer: str,
        *,
        context: str | None = None,
        latency_ms: float | None = None,
    ) -> "RAGRun":
        self._ensure_recording()
        if not isinstance(answer, str) or not answer.strip():
            raise InvalidAnswerError(
                "Required pipeline output 'answer' must be a non-empty string."
            )
        self._record.answer = answer.strip()
        if context is not None:
            self._record.supplied_context = str(context)
        if latency_ms is not None:
            self._record.generation_latency_ms = float(latency_ms)
        return self

    def record_failure(self, error: str | Exception) -> ExecutionRecord:
        """Record a pipeline failure without discarding already recorded telemetry."""
        self._ensure_recording()
        self._record.error = str(error)
        self._record.finished_at = utc_now_iso()
        self._record.status = RunStatus.FAILED
        return self._record

    def to_dict(self) -> dict[str, Any]:
        return self._record.to_dict()

    def finish(self, *, error: str | None = None) -> ExecutionRecord:
        self._ensure_recording()
        if error is not None:
            return self.record_failure(error)

        missing = []
        if self._record.retrieved_documents is None:
            missing.append("retrieval documents (call record_retrieval(...))")
        if self._record.answer is None:
            missing.append("generated answer (call record_generation(...))")
        if missing:
            raise MissingPipelineDataError(missing)

        self._record.finished_at = utc_now_iso()
        self._record.status = RunStatus.COMPLETED
        return self._record

    def _ensure_recording(self) -> None:
        if self._record.status is not RunStatus.RECORDING:
            raise RunStateError(
                f"Run '{self.run_id}' is already {self._record.status.value}; "
                f"start a new run to record another execution."
            )
