from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .results import AggregateMetrics, PerQueryResult


@dataclass
class EvaluationReport:
    per_query: list[PerQueryResult]
    aggregate: AggregateMetrics
    dataset_name: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "config": dict(self.config),
            "metadata": dict(self.metadata),
            "aggregate": self.aggregate.to_dict(),
            "per_query": [pq.to_dict() for pq in self.per_query],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        lines = []
        lines.append(f"Evaluation Report (dataset={self.dataset_name})")
        lines.append(f"Queries evaluated: {self.aggregate.num_evaluated}/{self.aggregate.num_queries}")
        if self.aggregate.retrieval:
            lines.append("Retrieval: " + ", ".join(f"{k}={v:.4f}" for k, v in sorted(self.aggregate.retrieval.items())))
        if self.aggregate.answer:
            lines.append("Answer: " + ", ".join(f"{k}={v:.4f}" for k, v in sorted(self.aggregate.answer.items())))
        if self.aggregate.faithfulness:
            lines.append("Faithfulness: " + ", ".join(f"{k}={v:.4f}" for k, v in sorted(self.aggregate.faithfulness.items())))
        if self.aggregate.abstention:
            lines.append("Abstention: " + ", ".join(f"{k}={v:.4f}" for k, v in sorted(self.aggregate.abstention.items())))
        if self.aggregate.latency:
            for stage, stats in self.aggregate.latency.items():
                if isinstance(stats, dict) and stats.get("mean_ms") is not None:
                    lines.append(f"Latency ({stage}): mean={stats['mean_ms']}ms, p95={stats['p95_ms']}ms (n={stats['count_valid']})")
        return "\n".join(lines)
