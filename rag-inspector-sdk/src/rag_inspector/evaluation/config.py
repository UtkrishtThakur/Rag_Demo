from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationConfig:
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    precision_denominator: str = "k"  # "k" (standard Cranfield) or "retrieved"
    compute_retrieval: bool = True
    compute_answer_correctness: bool = True
    compute_faithfulness: bool = True
    compute_abstention: bool = True
    compute_latency: bool = True
    match_by_id: bool = True
    match_by_text_overlap: bool = False
    ignore_missing_relevant: bool = True
    strict_missing_runs: bool = False
    include_details: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for k in self.k_values:
            if k <= 0:
                raise ValueError("k_values must be positive")
        if self.precision_denominator not in {"k", "retrieved"}:
            raise ValueError("precision_denominator must be either 'k' or 'retrieved'")
