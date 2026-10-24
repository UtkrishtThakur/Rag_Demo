"""Evaluation engine for RAG Inspector SDK."""

from .config import EvaluationConfig
from .dataset import EvaluationDataset, LabeledQuery, RelevanceJudgment
from .evaluator import (
    AbstentionEvaluator,
    AnswerCorrectnessEvaluator,
    Evaluator,
    FaithfulnessEvaluator,
    RetrievalEvaluator,
)
from .metrics import (
    average_precision_at_k,
    compute_latency_statistics,
    dcg_at_k,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rouge_l,
)
from .orchestrator import RAGEvaluator
from .report import EvaluationReport
from .results import AggregateMetrics, PerQueryResult

__all__ = [
    "EvaluationConfig",
    "EvaluationDataset",
    "LabeledQuery",
    "RelevanceJudgment",
    "Evaluator",
    "RetrievalEvaluator",
    "AnswerCorrectnessEvaluator",
    "FaithfulnessEvaluator",
    "AbstentionEvaluator",
    "RAGEvaluator",
    "EvaluationReport",
    "PerQueryResult",
    "AggregateMetrics",
    "recall_at_k",
    "precision_at_k",
    "mean_reciprocal_rank",
    "average_precision_at_k",
    "mean_average_precision",
    "dcg_at_k",
    "ndcg_at_k",
    "rouge_l",
    "compute_latency_statistics",
]
