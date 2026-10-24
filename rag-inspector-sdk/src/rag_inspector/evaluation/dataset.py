from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass
class RelevanceJudgment:
    query: str
    query_id: str | None = None
    relevant_doc_ids: set[str] = field(default_factory=set)
    relevant_docs: list[dict[str, Any]] = field(default_factory=list)
    relevance_scores: dict[str, float] = field(default_factory=dict)
    expected_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query must be a non-empty string")
        if self.query_id is not None and not isinstance(self.query_id, str):
            self.query_id = str(self.query_id).strip()
        if not isinstance(self.relevant_doc_ids, set):
            self.relevant_doc_ids = set(self.relevant_doc_ids) if self.relevant_doc_ids else set()


@dataclass
class LabeledQuery:
    query: str
    query_id: str | None = None
    relevant_doc_ids: set[str] = field(default_factory=set)
    expected_answer: str | None = None
    ground_truth_context: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query must be a non-empty string")
        if self.query_id is not None and not isinstance(self.query_id, str):
            self.query_id = str(self.query_id).strip()
        if not isinstance(self.relevant_doc_ids, set):
            self.relevant_doc_ids = set(self.relevant_doc_ids) if self.relevant_doc_ids else set()


class EvaluationDataset:
    def __init__(self, name: str = "default", metadata: dict[str, Any] | None = None):
        self.name = name
        self.metadata = metadata or {}
        self._judgments_by_query: dict[str, RelevanceJudgment] = {}
        self._judgments_by_id: dict[str, RelevanceJudgment] = {}
        self._labeled_by_query: dict[str, LabeledQuery] = {}
        self._labeled_by_id: dict[str, LabeledQuery] = {}

    @property
    def size(self) -> int:
        return len(self._judgments_by_query) or len(self._judgments_by_id)

    def add_judgment(self, judgment: RelevanceJudgment) -> "EvaluationDataset":
        if not isinstance(judgment, RelevanceJudgment):
            raise TypeError("judgment must be a RelevanceJudgment")
        q_key = judgment.query.strip()
        self._judgments_by_query[q_key] = judgment
        if judgment.query_id:
            self._judgments_by_id[judgment.query_id.strip()] = judgment
        return self

    def add_labeled_query(self, labeled: LabeledQuery) -> "EvaluationDataset":
        if not isinstance(labeled, LabeledQuery):
            raise TypeError("labeled must be a LabeledQuery")
        q_key = labeled.query.strip()
        self._labeled_by_query[q_key] = labeled
        if labeled.query_id:
            self._labeled_by_id[labeled.query_id.strip()] = labeled

        j = RelevanceJudgment(
            query=q_key,
            query_id=labeled.query_id,
            relevant_doc_ids=set(labeled.relevant_doc_ids),
            expected_answer=labeled.expected_answer,
            metadata=dict(labeled.metadata),
        )
        self.add_judgment(j)
        return self

    def get_relevant_ids(self, query: str, query_id: str | None = None) -> set[str]:
        j = self.get_judgment(query, query_id=query_id)
        if j:
            return set(j.relevant_doc_ids)
        return set()

    def get_expected_answer(self, query: str, query_id: str | None = None) -> str | None:
        j = self.get_judgment(query, query_id=query_id)
        if j and j.expected_answer:
            return j.expected_answer
        return None

    def get_judgment(self, query: str, query_id: str | None = None) -> RelevanceJudgment | None:
        if query_id and query_id.strip() in self._judgments_by_id:
            return self._judgments_by_id[query_id.strip()]
        q_key = query.strip() if isinstance(query, str) else str(query)
        return self._judgments_by_query.get(q_key)

    def queries(self) -> list[str]:
        return sorted(list(self._judgments_by_query.keys()))

    def query_ids(self) -> list[str]:
        return sorted(list(self._judgments_by_id.keys()))

    def to_dict(self) -> dict[str, Any]:
        items = []
        for q in self.queries():
            j = self.get_judgment(q)
            if j:
                items.append({
                    "query_id": j.query_id,
                    "query": j.query,
                    "relevant_doc_ids": sorted(list(j.relevant_doc_ids)),
                    "relevant_docs": j.relevant_docs,
                    "relevance_scores": dict(j.relevance_scores),
                    "expected_answer": j.expected_answer,
                    "metadata": dict(j.metadata),
                })
        return {"name": self.name, "metadata": dict(self.metadata), "items": items}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationDataset":
        ds = cls(name=data.get("name", "default"), metadata=data.get("metadata") or {})
        for item in data.get("items", []) or []:
            if not isinstance(item, dict):
                continue
            j = RelevanceJudgment(
                query=item.get("query", ""),
                query_id=item.get("query_id"),
                relevant_doc_ids=set(item.get("relevant_doc_ids") or []),
                relevant_docs=list(item.get("relevant_docs") or []),
                relevance_scores=dict(item.get("relevance_scores") or {}),
                expected_answer=item.get("expected_answer"),
                metadata=dict(item.get("metadata") or {}),
            )
            ds.add_judgment(j)
        return ds
