from __future__ import annotations

from typing import Any, Sequence

from ..schemas import ExecutionRecord
from .config import EvaluationConfig
from .dataset import EvaluationDataset
from .evaluator import (
    AbstentionEvaluator,
    AnswerCorrectnessEvaluator,
    FaithfulnessEvaluator,
    RetrievalEvaluator,
)
from .metrics import compute_latency_statistics
from .report import EvaluationReport
from .results import AggregateMetrics, PerQueryResult


def _get_exec_key(ex: ExecutionRecord) -> str:
    if ex.query_id and ex.query_id.strip():
        return ex.query_id.strip()
    return ex.question.strip()


class RAGEvaluator:
    def __init__(self, config: EvaluationConfig | None = None):
        self.config = config or EvaluationConfig()
        self.retrieval_eval = RetrievalEvaluator(self.config)
        self.answer_eval = AnswerCorrectnessEvaluator(self.config)
        self.faithfulness_eval = FaithfulnessEvaluator(self.config)
        self.abstention_eval = AbstentionEvaluator(self.config)

    def evaluate_executions(
        self,
        executions: Sequence[ExecutionRecord] | ExecutionRecord,
        dataset: EvaluationDataset | None = None,
    ) -> EvaluationReport:
        if isinstance(executions, ExecutionRecord):
            executions = [executions]
        executions = list(executions)

        # Index executions by query_id and question text
        exec_by_id: dict[str, ExecutionRecord] = {}
        exec_by_query: dict[str, ExecutionRecord] = {}
        for ex in executions:
            if ex.query_id and ex.query_id.strip():
                exec_by_id[ex.query_id.strip()] = ex
            exec_by_query[ex.question.strip()] = ex

        target_queries: list[str] = []
        if dataset is not None:
            target_queries = dataset.queries()
        else:
            target_queries = list(exec_by_query.keys())

        # Evaluate through active evaluators
        retrieval_results = self.retrieval_eval.evaluate(executions, dataset, self.config) if self.config.compute_retrieval else []
        retrieval_map = {r.query.strip(): r for r in retrieval_results}

        answer_results = self.answer_eval.evaluate(executions, dataset, self.config) if self.config.compute_answer_correctness else []
        answer_map = {r.query.strip(): r for r in answer_results}

        faith_results = self.faithfulness_eval.evaluate(executions, dataset, self.config) if self.config.compute_faithfulness else []
        faith_map = {r.query.strip(): r for r in faith_results}

        abstention_results = self.abstention_eval.evaluate(executions, dataset, self.config) if self.config.compute_abstention else []
        abstention_map = {r.query.strip(): r for r in abstention_results}

        per_query_results: list[PerQueryResult] = []

        for q in target_queries:
            q_clean = q.strip()
            # Match by question text or query_id if dataset has judgment
            ex = exec_by_query.get(q_clean)
            if ex is None and dataset is not None:
                j = dataset.get_judgment(q_clean)
                if j and j.query_id and j.query_id in exec_by_id:
                    ex = exec_by_id[j.query_id]

            if ex is None:
                if self.config.strict_missing_runs:
                    raise ValueError(f"No execution found for query in dataset: {q}")
                pq = PerQueryResult(query=q, errors=["missing_execution"])
                per_query_results.append(pq)
                continue

            pq = retrieval_map.get(q_clean) or PerQueryResult(
                query=q,
                query_id=ex.query_id,
                run_id=ex.run_id,
                answer=ex.answer,
                retrieval_latency_ms=ex.retrieval_latency_ms,
                generation_latency_ms=ex.generation_latency_ms,
            )

            if q_clean in answer_map:
                pq.answer_metrics.update(answer_map[q_clean].answer_metrics)
            if q_clean in faith_map:
                pq.faithfulness_metrics.update(faith_map[q_clean].faithfulness_metrics)
            if q_clean in abstention_map:
                pq.abstention.update(abstention_map[q_clean].abstention)

            if pq.run_id is None:
                pq.run_id = ex.run_id
            if pq.query_id is None:
                pq.query_id = ex.query_id
            if pq.answer is None:
                pq.answer = ex.answer
            if pq.retrieval_latency_ms is None:
                pq.retrieval_latency_ms = ex.retrieval_latency_ms
            if pq.generation_latency_ms is None:
                pq.generation_latency_ms = ex.generation_latency_ms

            per_query_results.append(pq)

        # Aggregate metrics
        agg = AggregateMetrics(num_queries=len(target_queries))
        evaluated = 0

        retrieval_aggs: dict[str, list[float]] = {}
        answer_aggs: dict[str, list[float]] = {}
        faith_aggs: dict[str, list[float]] = {}
        abstention_aggs: dict[str, list[float]] = {}

        retrieval_latencies: list[float | None] = []
        generation_latencies: list[float | None] = []

        for pq in per_query_results:
            if pq.errors:
                continue
            evaluated += 1

            for k, v in pq.retrieval_metrics.items():
                if isinstance(v, (int, float)):
                    retrieval_aggs.setdefault(k, []).append(float(v))

            for k, v in pq.answer_metrics.items():
                if isinstance(v, (int, float)):
                    answer_aggs.setdefault(k, []).append(float(v))

            for k, v in pq.faithfulness_metrics.items():
                if isinstance(v, (int, float)):
                    faith_aggs.setdefault(k, []).append(float(v))

            for k, v in pq.abstention.items():
                if isinstance(v, (int, float)):
                    abstention_aggs.setdefault(k, []).append(float(v))

            retrieval_latencies.append(pq.retrieval_latency_ms)
            generation_latencies.append(pq.generation_latency_ms)

        agg.num_evaluated = evaluated

        for k, vals in retrieval_aggs.items():
            if vals:
                agg.retrieval[k] = round(sum(vals) / len(vals), 4)

        for k, vals in answer_aggs.items():
            if vals:
                agg.answer[k] = round(sum(vals) / len(vals), 4)

        for k, vals in faith_aggs.items():
            if vals:
                agg.faithfulness[k] = round(sum(vals) / len(vals), 4)

        for k, vals in abstention_aggs.items():
            if vals:
                agg.abstention[k] = round(sum(vals) / len(vals), 4)

        if self.config.compute_latency:
            agg.latency = {
                "retrieval": compute_latency_statistics(retrieval_latencies),
                "generation": compute_latency_statistics(generation_latencies),
            }

        report = EvaluationReport(
            per_query=per_query_results,
            aggregate=agg,
            dataset_name=dataset.name if dataset is not None else None,
            config={
                "k_values": self.config.k_values,
                "precision_denominator": self.config.precision_denominator,
                "compute_retrieval": self.config.compute_retrieval,
                "compute_answer_correctness": self.config.compute_answer_correctness,
                "compute_faithfulness": self.config.compute_faithfulness,
                "compute_abstention": self.config.compute_abstention,
                "compute_latency": self.config.compute_latency,
            },
        )
        return report

    def evaluate_dataset(self, dataset: EvaluationDataset, executions: Sequence[ExecutionRecord]) -> EvaluationReport:
        return self.evaluate_executions(executions, dataset=dataset)

    def evaluate_single(self, execution: ExecutionRecord, dataset: EvaluationDataset | None = None) -> EvaluationReport:
        return self.evaluate_executions([execution], dataset=dataset)
