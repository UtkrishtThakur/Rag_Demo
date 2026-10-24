from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

VALID_ANSWER_CORRECTNESS = {"correct", "partially_correct", "incorrect", "unable_to_verify"}
VALID_RETRIEVAL_RELEVANCE = {"relevant", "partially_relevant", "irrelevant"}


@dataclass
class AnnotationRecord:
    query_id: str
    query: str
    rater_id: str
    answer_correctness: str | None = None
    retrieval_relevance: str | None = None
    sufficient_evidence_retrieved: bool | None = None
    answer_supported_by_context: bool | None = None
    answerable: bool | None = None
    primary_failure_type: str | None = None
    secondary_failure_types: list[str] = field(default_factory=list)
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.query_id, str) or not self.query_id.strip():
            raise ValueError("'query_id' must be a non-empty string.")
        if not isinstance(self.rater_id, str) or not self.rater_id.strip():
            raise ValueError("'rater_id' must be a non-empty string.")
        if not isinstance(self.query, str):
            raise ValueError("'query' must be a string.")
        if self.answer_correctness is not None and self.answer_correctness not in VALID_ANSWER_CORRECTNESS:
            raise ValueError(
                f"Invalid answer_correctness: '{self.answer_correctness}'. Allowed: {sorted(VALID_ANSWER_CORRECTNESS)}"
            )
        if self.retrieval_relevance is not None and self.retrieval_relevance not in VALID_RETRIEVAL_RELEVANCE:
            raise ValueError(
                f"Invalid retrieval_relevance: '{self.retrieval_relevance}'. Allowed: {sorted(VALID_RETRIEVAL_RELEVANCE)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AnnotationRecord":
        return cls(**d)


@dataclass
class AnnotationSet:
    name: str
    version: str = "1.0"
    annotations: list[AnnotationRecord] = field(default_factory=list)
    adjudications: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_annotation(self, record: AnnotationRecord) -> "AnnotationSet":
        if not isinstance(record, AnnotationRecord):
            raise TypeError("record must be an instance of AnnotationRecord.")
        self.annotations.append(record)
        return self

    def validate(self) -> dict[str, Any]:
        """Validate internal consistency: detect duplicate (query_id, rater_id), conflicts."""
        seen: dict[tuple[str, str], list[AnnotationRecord]] = defaultdict(list)
        for a in self.annotations:
            seen[(a.query_id, a.rater_id)].append(a)

        duplicates = [
            {"query_id": qid, "rater_id": rid, "count": len(recs)}
            for (qid, rid), recs in seen.items()
            if len(recs) > 1
        ]

        # Check conflicting annotations between different raters for same query
        by_query: dict[str, dict[str, str | None]] = defaultdict(dict)
        for a in self.annotations:
            by_query[a.query_id][a.rater_id] = a.primary_failure_type

        conflicts = []
        for qid, rater_labels in by_query.items():
            distinct_labels = {v for v in rater_labels.values() if v is not None}
            if len(distinct_labels) > 1:
                conflicts.append({"query_id": qid, "labels": rater_labels})

        return {
            "is_valid": len(duplicates) == 0,
            "total_annotations": len(self.annotations),
            "duplicates": duplicates,
            "conflicts": conflicts,
        }

    def get_consensus(self, attribute: str = "primary_failure_type") -> dict[str, Any]:
        """Compute majority-vote consensus per query_id with adjudication override."""
        by_query: dict[str, list[Any]] = defaultdict(list)
        for a in self.annotations:
            val = getattr(a, attribute, None)
            if val is not None:
                by_query[a.query_id].append(val)

        consensus: dict[str, Any] = {}
        unresolved: list[str] = []

        for qid, vals in by_query.items():
            if qid in self.adjudications and attribute in self.adjudications[qid]:
                consensus[qid] = self.adjudications[qid][attribute]
                continue

            from collections import Counter

            counts = Counter(vals).most_common()
            if not counts:
                continue
            if len(counts) > 1 and counts[0][1] == counts[1][1]:
                # Tie
                unresolved.append(qid)
                consensus[qid] = counts[0][0]  # Tentative majority
            else:
                consensus[qid] = counts[0][0]

        return {
            "attribute": attribute,
            "consensus": consensus,
            "unresolved_ties": unresolved,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "annotations": [a.to_dict() for a in self.annotations],
            "adjudications": dict(self.adjudications),
            "metadata": dict(self.metadata),
        }

    def save_jsonl(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for a in self.annotations:
                f.write(json.dumps(a.to_dict()) + "\n")
            if self.adjudications:
                f.write(json.dumps({"_type": "adjudications", "data": self.adjudications}) + "\n")

    @classmethod
    def load_jsonl(cls, path: str | Path, name: str = "imported") -> "AnnotationSet":
        path = Path(path)
        annotations = []
        adjudications = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("_type") == "adjudications":
                    adjudications.update(obj.get("data") or {k: v for k, v in obj.items() if k != "_type"})
                    continue
                annotations.append(AnnotationRecord.from_dict(obj))
        return cls(name=name, annotations=annotations, adjudications=adjudications)
