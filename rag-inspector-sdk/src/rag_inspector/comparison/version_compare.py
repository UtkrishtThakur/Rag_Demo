from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..schemas import ExecutionRecord


@dataclass
class VersionComparison:
    version_a: str
    version_b: str
    paired: bool = True
    metrics_a: dict[str, float] = field(default_factory=dict)
    metrics_b: dict[str, float] = field(default_factory=dict)
    deltas: dict[str, float] = field(default_factory=dict)
    relative_changes: dict[str, float | None] = field(default_factory=dict)
    per_query: list[dict[str, Any]] = field(default_factory=list)
    regressions: list[dict[str, Any]] = field(default_factory=list)
    improvements: list[dict[str, Any]] = field(default_factory=list)
    missing_matches: list[str] = field(default_factory=list)
    bootstrap_ci: dict[str, tuple[float, float]] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_a": self.version_a,
            "version_b": self.version_b,
            "paired": self.paired,
            "metrics_a": dict(self.metrics_a),
            "metrics_b": dict(self.metrics_b),
            "deltas": dict(self.deltas),
            "relative_changes": {
                k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in self.relative_changes.items()
            },
            "per_query": list(self.per_query),
            "regressions": list(self.regressions),
            "improvements": list(self.improvements),
            "missing_matches": list(self.missing_matches),
            "bootstrap_ci": {
                k: [round(x, 6) for x in v] if v is not None else None
                for k, v in self.bootstrap_ci.items()
            },
            "summary": self.summary,
        }


def _paired_bootstrap_ci(
    diffs: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> tuple[float, float] | None:
    if not diffs:
        return None
    rng = random.Random(seed)
    n = len(diffs)
    samples = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += rng.choice(diffs)
        samples.append(s / n)
    samples.sort()
    lo = samples[int((alpha / 2) * n_boot)]
    hi = samples[int((1 - alpha / 2) * n_boot)]
    return (float(lo), float(hi))


def _extract_query_key(ex: ExecutionRecord) -> str:
    if ex.query_id and ex.query_id.strip():
        return ex.query_id.strip()
    return ex.question.strip()


def compare_versions(
    execs_a: Sequence[ExecutionRecord],
    execs_b: Sequence[ExecutionRecord],
    version_a: str = "v1",
    version_b: str = "v2",
    dataset: Any = None,
    regression_thresholds: dict[str, float] | None = None,
    lower_is_better: set[str] | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> VersionComparison:
    """Compare two RAG pipeline versions across paired executions."""
    lower_is_better = lower_is_better or {"latency_p95_ms", "latency_mean_ms", "retrieval_latency_ms", "generation_latency_ms", "error_rate"}
    thresholds = regression_thresholds or {
        "recall@5": 0.05,
        "exact_match": 0.05,
        "token_f1": 0.05,
        "faithfulness": 0.05,
        "latency_p95_ms": 10.0,
        "latency_mean_ms": 10.0,
    }

    # Match by query_id or question
    map_a: dict[str, ExecutionRecord] = {}
    for ex in execs_a:
        key = _extract_query_key(ex)
        map_a[key] = ex

    map_b: dict[str, ExecutionRecord] = {}
    for ex in execs_b:
        key = _extract_query_key(ex)
        map_b[key] = ex

    common_keys = sorted(set(map_a.keys()) & set(map_b.keys()))
    missing_in_b = sorted(set(map_a.keys()) - set(map_b.keys()))
    missing_in_a = sorted(set(map_b.keys()) - set(map_a.keys()))
    missing_matches = sorted(set(missing_in_a) | set(missing_in_b))

    comp = VersionComparison(
        version_a=version_a,
        version_b=version_b,
        paired=True,
        missing_matches=missing_matches,
    )

    # Evaluate executions if dataset provided, or compute operational metrics
    eval_report_a = None
    eval_report_b = None
    if dataset is not None:
        try:
            from ..evaluation.orchestrator import RAGEvaluator

            evaluator = RAGEvaluator()
            eval_report_a = evaluator.evaluate_executions(execs_a, dataset=dataset)
            eval_report_b = evaluator.evaluate_executions(execs_b, dataset=dataset)
        except Exception:
            eval_report_a = None
            eval_report_b = None

    # Helper to get per-query metrics for an execution
    def get_query_metrics(ex: ExecutionRecord, report: Any) -> dict[str, float]:
        metrics: dict[str, float] = {
            "answered": 1.0 if ex.answer and ex.answer.strip() else 0.0,
            "has_retrieval": 1.0 if ex.retrieved_documents else 0.0,
            "num_retrieved": float(len(ex.retrieved_documents or [])),
            "is_error": 1.0 if (ex.error or ex.status.value == "failed") else 0.0,
        }
        if ex.retrieval_latency_ms is not None:
            metrics["retrieval_latency_ms"] = float(ex.retrieval_latency_ms)
        if ex.generation_latency_ms is not None:
            metrics["generation_latency_ms"] = float(ex.generation_latency_ms)

        if report is not None:
            q_key = ex.question.strip()
            for pq in report.per_query:
                if pq.query.strip() == q_key:
                    for k, v in pq.retrieval_metrics.items():
                        metrics[k] = float(v)
                    for k, v in pq.answer_metrics.items():
                        metrics[k] = float(v)
                    for k, v in pq.faithfulness_metrics.items():
                        metrics[k] = float(v)
                    break
        elif dataset is not None:
            # Fallback simple answer match if report wasn't generated
            exp = dataset.get_expected_answer(ex.question)
            if exp and ex.answer:
                metrics["exact_match"] = 1.0 if ex.answer.strip().lower() == exp.strip().lower() else 0.0
        return metrics

    # Build per-query metrics
    per_query_diffs: dict[str, list[float]] = {}
    per_query_rows: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []

    for qk in common_keys:
        ex_a = map_a[qk]
        ex_b = map_b[qk]
        m_a = get_query_metrics(ex_a, eval_report_a)
        m_b = get_query_metrics(ex_b, eval_report_b)

        q_deltas: dict[str, float] = {}
        all_metric_keys = sorted(set(m_a.keys()) | set(m_b.keys()))

        for mk in all_metric_keys:
            val_a = m_a.get(mk, 0.0)
            val_b = m_b.get(mk, 0.0)
            diff = val_b - val_a
            q_deltas[mk] = diff
            per_query_diffs.setdefault(mk, []).append(diff)

            thresh = thresholds.get(mk, 0.01)
            is_lower = mk in lower_is_better

            # Check regression
            is_regression = (diff > thresh) if is_lower else (diff < -thresh)
            is_improvement = (diff < -thresh) if is_lower else (diff > thresh)

            if is_regression:
                regressions.append({
                    "query": qk,
                    "metric": mk,
                    "version_a": val_a,
                    "version_b": val_b,
                    "delta": diff,
                })
            elif is_improvement:
                improvements.append({
                    "query": qk,
                    "metric": mk,
                    "version_a": val_a,
                    "version_b": val_b,
                    "delta": diff,
                })

        per_query_rows.append({
            "query": qk,
            "metrics_a": m_a,
            "metrics_b": m_b,
            "deltas": q_deltas,
        })

    comp.per_query = per_query_rows
    comp.regressions = regressions
    comp.improvements = improvements

    # Aggregate metrics
    all_metrics: set[str] = set()
    for row in per_query_rows:
        all_metrics.update(row["metrics_a"].keys())
        all_metrics.update(row["metrics_b"].keys())

    metrics_a: dict[str, float] = {}
    metrics_b: dict[str, float] = {}
    deltas: dict[str, float] = {}
    rel_changes: dict[str, float | None] = {}
    bootstrap_cis: dict[str, tuple[float, float]] = {}

    for mk in sorted(all_metrics):
        vals_a = [row["metrics_a"][mk] for row in per_query_rows if mk in row["metrics_a"]]
        vals_b = [row["metrics_b"][mk] for row in per_query_rows if mk in row["metrics_b"]]

        mean_a = sum(vals_a) / len(vals_a) if vals_a else 0.0
        mean_b = sum(vals_b) / len(vals_b) if vals_b else 0.0

        metrics_a[mk] = mean_a
        metrics_b[mk] = mean_b
        delta = mean_b - mean_a
        deltas[mk] = delta

        # Relative change is undefined if baseline mean_a == 0.0
        if mean_a != 0.0:
            rel_changes[mk] = delta / abs(mean_a)
        else:
            rel_changes[mk] = None

        diffs = per_query_diffs.get(mk, [])
        if len(diffs) >= 2:
            ci = _paired_bootstrap_ci(diffs, n_boot=n_boot, alpha=alpha, seed=seed)
            if ci is not None:
                bootstrap_cis[mk] = ci

    comp.metrics_a = metrics_a
    comp.metrics_b = metrics_b
    comp.deltas = deltas
    comp.relative_changes = rel_changes
    comp.bootstrap_ci = bootstrap_cis

    num_paired = len(common_keys)
    num_unmatched = len(missing_matches)
    comp.summary = (
        f"Compared {version_a} vs {version_b}: {num_paired} paired queries, "
        f"{num_unmatched} unmatched ({len(missing_in_b)} missing in {version_b}, "
        f"{len(missing_in_a)} missing in {version_a}). "
        f"Detected {len(regressions)} metric regressions and {len(improvements)} improvements."
    )

    return comp
