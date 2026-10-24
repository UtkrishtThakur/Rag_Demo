from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PerQueryResult:
    query: str
    query_id: str | None = None
    run_id: str | None = None
    retrieval_metrics: dict[str, float] = field(default_factory=dict)
    answer_metrics: dict[str, Any] = field(default_factory=dict)
    faithfulness_metrics: dict[str, float] = field(default_factory=dict)
    abstention: dict[str, Any] = field(default_factory=dict)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    retrieved_texts: list[str] = field(default_factory=list)
    answer: str | None = None
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    errors: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "query_id": self.query_id,
            "run_id": self.run_id,
            "retrieval_metrics": dict(self.retrieval_metrics),
            "answer_metrics": dict(self.answer_metrics),
            "faithfulness_metrics": dict(self.faithfulness_metrics),
            "abstention": dict(self.abstention),
            "retrieved_doc_ids": list(self.retrieved_doc_ids),
            "retrieved_texts": list(self.retrieved_texts),
            "answer": self.answer,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "generation_latency_ms": self.generation_latency_ms,
            "errors": list(self.errors),
            "details": dict(self.details),
        }


@dataclass
class AggregateMetrics:
    num_queries: int = 0
    num_evaluated: int = 0
    retrieval: dict[str, float] = field(default_factory=dict)
    answer: dict[str, float] = field(default_factory=dict)
    faithfulness: dict[str, float] = field(default_factory=dict)
    abstention: dict[str, float] = field(default_factory=dict)
    latency: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_queries": self.num_queries,
            "num_evaluated": self.num_evaluated,
            "retrieval": dict(self.retrieval),
            "answer": dict(self.answer),
            "faithfulness": dict(self.faithfulness),
            "abstention": dict(self.abstention),
            "latency": dict(self.latency),
            "metadata": dict(self.metadata),
        }
