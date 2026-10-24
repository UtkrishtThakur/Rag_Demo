from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..evaluation.orchestrator import RAGEvaluator


@dataclass
class BenchmarkResult:
    benchmark_mode: str  # "evaluation_only" or "end_to_end"
    num_questions: int
    duration_seconds: float
    duration_minutes: float
    target_minutes: float = 15.0
    met_target: bool = False
    throughput_qps: float = 0.0
    per_stage_latency: dict[str, Any] = field(default_factory=dict)
    success_count: int = 0
    failed_count: int = 0
    timeout_count: int = 0
    errors: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_mode": self.benchmark_mode,
            "num_questions": self.num_questions,
            "duration_seconds": round(self.duration_seconds, 4),
            "duration_minutes": round(self.duration_minutes, 4),
            "target_minutes": self.target_minutes,
            "met_target": self.met_target,
            "throughput_qps": round(self.throughput_qps, 2),
            "per_stage_latency": dict(self.per_stage_latency),
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "timeout_count": self.timeout_count,
            "errors": list(self.errors),
            "details": dict(self.details),
        }


class BenchmarkRunner:
    """Benchmark SDK evaluation throughput or end-to-end RAG pipeline throughput."""

    def __init__(self, evaluator: RAGEvaluator | None = None):
        self.evaluator = evaluator or RAGEvaluator()

    def run_benchmark(
        self,
        executions: Sequence[Any],
        dataset: Any = None,
        target_minutes: float = 15.0,
    ) -> BenchmarkResult:
        """Run an SDK evaluation-only benchmark over previously recorded execution data."""
        start = time.perf_counter()
        errors: list[str] = []
        success = 0
        failed = 0
        per_stage: dict[str, Any] = {}

        try:
            report = self.evaluator.evaluate_executions(executions, dataset=dataset)
            success = report.aggregate.num_evaluated
            failed = report.aggregate.num_queries - report.aggregate.num_evaluated
            if report.aggregate.latency:
                per_stage = report.aggregate.latency
        except Exception as e:
            errors.append(str(e))
            failed = len(executions) if hasattr(executions, "__len__") else 0

        end = time.perf_counter()
        duration = end - start
        duration_min = duration / 60.0
        n_queries = len(executions) if hasattr(executions, "__len__") else 0
        qps = (n_queries / duration) if duration > 0 else 0.0

        return BenchmarkResult(
            benchmark_mode="evaluation_only",
            num_questions=n_queries,
            duration_seconds=duration,
            duration_minutes=duration_min,
            target_minutes=target_minutes,
            met_target=(duration_min <= target_minutes),
            throughput_qps=qps,
            per_stage_latency=per_stage,
            success_count=success,
            failed_count=failed,
            errors=errors,
            details={
                "note": "Evaluation-only benchmark measures SDK engine throughput over in-memory execution records. It does not measure external LLM or vector DB latency.",
            },
        )

    def run_end_to_end_benchmark(
        self,
        pipeline_fn: Callable[[str], dict[str, Any]],
        questions: Sequence[str],
        dataset: Any = None,
        target_minutes: float = 15.0,
        timeout_seconds_per_query: float = 30.0,
    ) -> BenchmarkResult:
        """Run an end-to-end RAG benchmark executing live retrieval, generation, and SDK evaluation."""
        start = time.perf_counter()
        errors: list[str] = []
        executions = []
        success = 0
        failed = 0
        timeouts = 0

        retrieval_times: list[float] = []
        generation_times: list[float] = []

        for q in questions:
            t0 = time.perf_counter()
            try:
                out = pipeline_fn(q)
                elapsed = time.perf_counter() - t0
                if elapsed > timeout_seconds_per_query:
                    timeouts += 1
                if "rag_inspector" in out:
                    from ..schemas import ExecutionRecord

                    executions.append(ExecutionRecord.from_dict(out["rag_inspector"]))
                    success += 1
                else:
                    success += 1
            except Exception as e:
                failed += 1
                errors.append(f"Query '{q}': {e}")

        # Now evaluate collected executions if dataset provided
        if executions and dataset is not None:
            try:
                self.evaluator.evaluate_executions(executions, dataset=dataset)
            except Exception as exc:
                errors.append(f"Evaluation stage error: {exc}")

        end = time.perf_counter()
        duration = end - start
        duration_min = duration / 60.0
        n_queries = len(questions)
        qps = (n_queries / duration) if duration > 0 else 0.0

        return BenchmarkResult(
            benchmark_mode="end_to_end",
            num_questions=n_queries,
            duration_seconds=duration,
            duration_minutes=duration_min,
            target_minutes=target_minutes,
            met_target=(duration_min <= target_minutes),
            throughput_qps=qps,
            per_stage_latency={
                "mean_pipeline_latency_ms": round((duration / n_queries) * 1000, 2) if n_queries > 0 else 0.0,
            },
            success_count=success,
            failed_count=failed,
            timeout_count=timeouts,
            errors=errors,
            details={
                "note": "End-to-end benchmark includes live retrieval and generation latencies.",
            },
        )
