from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Sequence

from ..schemas import ExecutionRecord
from .config import EvaluationConfig
from .dataset import EvaluationDataset
from .metrics import (
    average_precision_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rouge_l,
)
from .results import PerQueryResult

_COMMON_ABSTENTION_PHRASES = {
    "i don't know",
    "i do not know",
    "cannot answer",
    "can not answer",
    "unable to answer",
    "not enough information",
    "no relevant context",
    "insufficient context",
    "does not mention",
    "not mentioned in the provided",
}


def _is_abstention_text(text: str | None) -> bool:
    if not text or not text.strip():
        return True
    lower = text.strip().lower()
    for phrase in _COMMON_ABSTENTION_PHRASES:
        if phrase in lower:
            return True
    return False


def _tokenize_set(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))


class Evaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        executions: Sequence[ExecutionRecord] | ExecutionRecord,
        dataset: EvaluationDataset | None = None,
        config: EvaluationConfig | None = None,
    ) -> list[PerQueryResult]:
        raise NotImplementedError


class RetrievalEvaluator(Evaluator):
    def __init__(self, config: EvaluationConfig | None = None):
        self.config = config or EvaluationConfig()

    def evaluate(
        self,
        executions: Sequence[ExecutionRecord] | ExecutionRecord,
        dataset: EvaluationDataset | None = None,
        config: EvaluationConfig | None = None,
    ) -> list[PerQueryResult]:
        cfg = config or self.config
        if isinstance(executions, ExecutionRecord):
            executions = [executions]
        results: list[PerQueryResult] = []
        for ex in executions:
            results.append(self._eval_single(ex, dataset, cfg))
        return results

    def _eval_single(
        self,
        ex: ExecutionRecord,
        dataset: EvaluationDataset | None,
        cfg: EvaluationConfig,
    ) -> PerQueryResult:
        query = ex.question
        qid = ex.query_id
        retrieved_docs = ex.retrieved_documents or []

        retrieved_ids: list[str] = []
        retrieved_texts: list[str] = []
        for d in retrieved_docs:
            doc_id = getattr(d, "id", None)
            if doc_id is None and isinstance(d, dict):
                doc_id = d.get("id")
            if doc_id is not None and str(doc_id).strip():
                retrieved_ids.append(str(doc_id).strip())

            doc_text = getattr(d, "text", "") if hasattr(d, "text") else str(d.get("text", "") if isinstance(d, dict) else "")
            retrieved_texts.append(doc_text)

        qres = PerQueryResult(
            query=query,
            query_id=qid,
            run_id=ex.run_id,
            retrieved_doc_ids=retrieved_ids,
            retrieved_texts=retrieved_texts,
            answer=ex.answer,
            retrieval_latency_ms=ex.retrieval_latency_ms,
            generation_latency_ms=ex.generation_latency_ms,
        )

        if dataset is None:
            return qres

        relevant = dataset.get_relevant_ids(query, query_id=qid)
        judgment = dataset.get_judgment(query, query_id=qid)

        if not relevant and cfg.ignore_missing_relevant:
            qres.details["note"] = "no_relevant_annotations"
            return qres

        # Determine relevance scores for nDCG
        relevance_scores: dict[str, float] = {}
        if judgment and judgment.relevance_scores:
            relevance_scores = {str(k).strip(): float(v) for k, v in judgment.relevance_scores.items()}
        elif relevant:
            # Binary relevance fallback
            relevance_scores = {rid: 1.0 for rid in relevant}

        retrieval_metrics: dict[str, float] = {}
        for k in cfg.k_values:
            retrieval_metrics[f"recall@{k}"] = round(recall_at_k(retrieved_ids, relevant, k), 4)
            retrieval_metrics[f"precision@{k}"] = round(
                precision_at_k(retrieved_ids, relevant, k, denominator=cfg.precision_denominator),
                4,
            )
            if relevance_scores:
                retrieval_metrics[f"ndcg@{k}"] = round(ndcg_at_k(retrieved_ids, relevance_scores, k), 4)

        retrieval_metrics["mrr"] = round(mean_reciprocal_rank(retrieved_ids, relevant), 4)
        retrieval_metrics["mrr@10"] = round(mean_reciprocal_rank(retrieved_ids, relevant, k=10), 4)
        retrieval_metrics["map"] = round(
            average_precision_at_k(retrieved_ids, relevant, len(retrieved_ids)) if retrieved_ids else 0.0,
            4,
        )

        qres.retrieval_metrics = retrieval_metrics
        return qres


class AnswerCorrectnessEvaluator(Evaluator):
    def __init__(self, config: EvaluationConfig | None = None):
        self.config = config or EvaluationConfig()

    def evaluate(
        self,
        executions: Sequence[ExecutionRecord] | ExecutionRecord,
        dataset: EvaluationDataset | None = None,
        config: EvaluationConfig | None = None,
    ) -> list[PerQueryResult]:
        if isinstance(executions, ExecutionRecord):
            executions = [executions]
        results: list[PerQueryResult] = []

        for ex in executions:
            qres = PerQueryResult(
                query=ex.question,
                query_id=ex.query_id,
                run_id=ex.run_id,
                answer=ex.answer,
            )
            if dataset is None or not ex.answer:
                qres.answer_metrics["verdict"] = "unable_to_verify"
                qres.details["note"] = "missing_dataset_or_answer"
                results.append(qres)
                continue

            expected = dataset.get_expected_answer(ex.question, query_id=ex.query_id)
            if not expected:
                qres.answer_metrics["verdict"] = "unable_to_verify"
                qres.details["note"] = "no_expected_answer"
                results.append(qres)
                continue

            pred = ex.answer.strip()
            exp = expected.strip()
            exact = 1.0 if pred.lower() == exp.lower() else 0.0

            pred_tokens = _tokenize_set(pred)
            exp_tokens = _tokenize_set(exp)

            if not exp_tokens:
                token_f1 = 1.0 if pred == exp else 0.0
                overlap_ratio = 1.0 if pred == exp else 0.0
            else:
                overlap = len(pred_tokens & exp_tokens)
                token_f1 = (2 * overlap / (len(pred_tokens) + len(exp_tokens))) if (len(pred_tokens) + len(exp_tokens)) > 0 else 0.0
                overlap_ratio = overlap / len(exp_tokens)

            # LCS-based ROUGE-L
            rouge_res = rouge_l(pred, exp)

            # Assign categorical verdict
            if exact == 1.0 or token_f1 >= 0.85 or rouge_res["f1"] >= 0.85:
                verdict = "correct"
            elif token_f1 >= 0.40 or rouge_res["f1"] >= 0.40:
                verdict = "partially_correct"
            else:
                verdict = "incorrect"

            qres.answer_metrics = {
                "verdict": verdict,
                "exact_match": exact,
                "token_f1": round(token_f1, 4),
                "rouge_l_f1": rouge_res["f1"],
                "rouge_l_precision": rouge_res["precision"],
                "rouge_l_recall": rouge_res["recall"],
                "token_overlap_ratio": round(overlap_ratio, 4),
            }
            results.append(qres)

        return results


class FaithfulnessEvaluator(Evaluator):
    def __init__(self, config: EvaluationConfig | None = None):
        self.config = config or EvaluationConfig()

    def evaluate(
        self,
        executions: Sequence[ExecutionRecord] | ExecutionRecord,
        dataset: EvaluationDataset | None = None,
        config: EvaluationConfig | None = None,
    ) -> list[PerQueryResult]:
        if isinstance(executions, ExecutionRecord):
            executions = [executions]
        results: list[PerQueryResult] = []

        for ex in executions:
            qres = PerQueryResult(
                query=ex.question,
                query_id=ex.query_id,
                run_id=ex.run_id,
                answer=ex.answer,
            )
            if not ex.answer:
                qres.faithfulness_metrics["faithfulness"] = 0.0
                results.append(qres)
                continue

            # Prefer supplied_context if recorded, otherwise join retrieved documents
            context_text = ex.supplied_context
            if not context_text:
                texts = []
                for d in (ex.retrieved_documents or []):
                    t = getattr(d, "text", "") if hasattr(d, "text") else str(d.get("text", "") if isinstance(d, dict) else "")
                    if t:
                        texts.append(t)
                context_text = " ".join(texts)

            if not context_text.strip():
                qres.faithfulness_metrics["faithfulness"] = 0.0
                qres.details["note"] = "no_context"
                results.append(qres)
                continue

            ans_tokens = _tokenize_set(ex.answer)
            ctx_tokens = _tokenize_set(context_text)

            if not ans_tokens:
                qres.faithfulness_metrics["faithfulness"] = 1.0
                results.append(qres)
                continue

            supported_tokens = len(ans_tokens & ctx_tokens)
            token_ratio = supported_tokens / len(ans_tokens)

            # Sentence-level support ratio
            sentences = [s.strip() for s in re.split(r"[.!?]+", ex.answer) if s.strip()]
            supported_sentences = 0
            for sent in sentences:
                stoks = _tokenize_set(sent)
                if stoks and (len(stoks & ctx_tokens) / len(stoks)) >= 0.5:
                    supported_sentences += 1
            sent_ratio = (supported_sentences / len(sentences)) if sentences else 1.0

            qres.faithfulness_metrics = {
                "faithfulness": round(token_ratio, 4),
                "lexical_token_support": round(token_ratio, 4),
                "sentence_support_ratio": round(sent_ratio, 4),
            }
            results.append(qres)

        return results


class AbstentionEvaluator(Evaluator):
    def __init__(self, config: EvaluationConfig | None = None):
        self.config = config or EvaluationConfig()

    def evaluate(
        self,
        executions: Sequence[ExecutionRecord] | ExecutionRecord,
        dataset: EvaluationDataset | None = None,
        config: EvaluationConfig | None = None,
    ) -> list[PerQueryResult]:
        if isinstance(executions, ExecutionRecord):
            executions = [executions]
        results: list[PerQueryResult] = []

        for ex in executions:
            qres = PerQueryResult(
                query=ex.question,
                query_id=ex.query_id,
                run_id=ex.run_id,
                answer=ex.answer,
            )
            is_abstained = _is_abstention_text(ex.answer)
            judgment = dataset.get_judgment(ex.question, query_id=ex.query_id) if dataset else None

            is_answerable: bool | None = None
            if judgment is not None:
                is_answerable = getattr(judgment, "metadata", {}).get("answerable")
                if is_answerable is None and judgment.relevant_doc_ids:
                    is_answerable = True

            abstention_data: dict[str, Any] = {
                "is_abstention": is_abstained,
                "coverage": 0.0 if is_abstained else 1.0,
                "annotated_answerable": is_answerable,
            }

            if is_answerable is False:
                abstention_data["correct_abstention"] = 1.0 if is_abstained else 0.0
            elif is_answerable is True:
                abstention_data["unnecessary_abstention"] = 1.0 if is_abstained else 0.0

            qres.abstention = abstention_data
            results.append(qres)

        return results
