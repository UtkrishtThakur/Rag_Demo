from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class RunStatus(str, Enum):
    RECORDING = "recording"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class DocumentRecord:
    text: str
    id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None
    rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "text": self.text,
            "metadata": dict(self.metadata),
        }
        if self.score is not None:
            result["score"] = self.score
        if self.rank is not None:
            result["rank"] = self.rank
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentRecord":
        return cls(
            text=str(data.get("text", "")),
            id=data.get("id"),
            metadata=dict(data.get("metadata") or {}),
            score=data.get("score"),
            rank=data.get("rank"),
        )


@dataclass
class ExecutionRecord:
    run_id: str
    question: str
    started_at: str
    query_id: str | None = None
    retrieved_documents: list[DocumentRecord] | None = None
    supplied_context: str | None = None
    answer: str | None = None
    finished_at: str | None = None
    status: RunStatus = RunStatus.RECORDING
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    rag_version: str | None = None
    dataset_id: str | None = None
    dataset_version: str | None = None
    annotation_version: str | None = None
    reference_answer: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "query_id": self.query_id,
            "question": self.question,
            "started_at": self.started_at,
            "retrieved_documents": (
                None if self.retrieved_documents is None
                else [document.to_dict() for document in self.retrieved_documents]
            ),
            "supplied_context": self.supplied_context,
            "answer": self.answer,
            "finished_at": self.finished_at,
            "status": self.status.value,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "generation_latency_ms": self.generation_latency_ms,
            "rag_version": self.rag_version,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "annotation_version": self.annotation_version,
            "reference_answer": self.reference_answer,
            "error": self.error,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionRecord":
        retrieved_raw = data.get("retrieved_documents")
        retrieved_docs = (
            [DocumentRecord.from_dict(d) for d in retrieved_raw]
            if retrieved_raw is not None
            else None
        )
        status_val = data.get("status", RunStatus.RECORDING.value)
        try:
            status = RunStatus(status_val)
        except ValueError:
            status = RunStatus.RECORDING

        return cls(
            run_id=data.get("run_id", ""),
            query_id=data.get("query_id"),
            question=data.get("question", ""),
            started_at=data.get("started_at", utc_now_iso()),
            retrieved_documents=retrieved_docs,
            supplied_context=data.get("supplied_context"),
            answer=data.get("answer"),
            finished_at=data.get("finished_at"),
            status=status,
            retrieval_latency_ms=data.get("retrieval_latency_ms"),
            generation_latency_ms=data.get("generation_latency_ms"),
            rag_version=data.get("rag_version"),
            dataset_id=data.get("dataset_id"),
            dataset_version=data.get("dataset_version"),
            annotation_version=data.get("annotation_version"),
            reference_answer=data.get("reference_answer"),
            error=data.get("error"),
            metadata=dict(data.get("metadata") or {}),
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
