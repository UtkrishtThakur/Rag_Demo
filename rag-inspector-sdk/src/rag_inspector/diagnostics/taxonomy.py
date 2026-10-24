from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FailureType(str, Enum):
    NO_FAILURE = "no_failure"
    WRONG_OR_IRRELEVANT_RETRIEVAL = "wrong_or_irrelevant_retrieval"
    MISSING_RELEVANT_EVIDENCE = "missing_relevant_evidence"
    IGNORED_CONTEXT = "ignored_context"
    UNSUPPORTED_HALLUCINATION = "unsupported_hallucination"
    STALE_SOURCE = "stale_source"
    CONFLICTING_EVIDENCE_MISHANDLED = "conflicting_evidence_mishandled"
    INCOMPLETE_OR_INCORRECT_GENERATION = "incomplete_or_incorrect_generation"
    INCORRECT_ABSTENTION = "incorrect_abstention"
    FAILURE_TO_ABSTAIN = "failure_to_abstain"
    UNKNOWN = "unknown"


@dataclass
class FailureDiagnosis:
    primary: FailureType
    secondary: list[FailureType] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    confidence: float | None = None
    unclassified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary.value,
            "secondary": [s.value for s in self.secondary],
            "evidence": dict(self.evidence),
            "explanation": self.explanation,
            "confidence": self.confidence,
            "unclassified": self.unclassified,
        }


_STOP_WORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "is", "was", "are", "were", "it", "this", "that", "be", "by", "as", "from",
}


def _tokenize(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))


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


def _is_abstention(text: str | None) -> bool:
    if not text or not text.strip():
        return True
    lower = text.strip().lower()
    for phrase in _COMMON_ABSTENTION_PHRASES:
        if phrase in lower:
            return True
    return False


class FailureClassifier:
    """Evidence-based diagnostic classifier for RAG executions."""

    def classify(self, execution: Any, dataset: Any = None, context: dict[str, Any] | None = None) -> FailureDiagnosis:
        ctx = context or {}
        retrieved_docs = getattr(execution, "retrieved_documents", None) or []
        answer = getattr(execution, "answer", None)
        query = getattr(execution, "question", "")
        supplied_context = getattr(execution, "supplied_context", None)

        relevant_set: set[str] = set()
        expected_answer: str | None = None
        judgment = None
        if dataset is not None:
            try:
                relevant_set = dataset.get_relevant_ids(query)
                expected_answer = dataset.get_expected_answer(query)
                judgment = dataset.get_judgment(query) if hasattr(dataset, "get_judgment") else None
            except Exception:
                relevant_set = set()

        # Check answerability annotations if present in judgment or context
        is_answerable = ctx.get("answerable")
        if is_answerable is None and judgment is not None:
            is_answerable = getattr(judgment, "metadata", {}).get("answerable")

        # Collect basic evidence
        retrieved_ids = []
        context_texts = []
        for d in retrieved_docs:
            did = getattr(d, "id", None)
            if did is None and isinstance(d, dict):
                did = d.get("id")
            if did is not None and str(did).strip():
                retrieved_ids.append(str(did).strip())
            d_text = getattr(d, "text", "") if hasattr(d, "text") else str(d.get("text", "") if isinstance(d, dict) else "")
            if d_text:
                context_texts.append(d_text)

        retrieved_set = set(retrieved_ids)
        has_answer = bool(answer and answer.strip())
        is_abstained = _is_abstention(answer)

        # Full text of context available
        full_context_text = supplied_context if supplied_context else " ".join(context_texts)

        evidence: dict[str, Any] = {
            "num_retrieved": len(retrieved_docs),
            "retrieved_ids": list(retrieved_ids),
            "num_relevant_annotated": len(relevant_set),
            "relevant_ids": sorted(list(relevant_set)),
            "has_answer": has_answer,
            "is_abstention": is_abstained,
        }

        # Rule 1: Explicitly unanswerable query policy
        if is_answerable is False:
            if has_answer and not is_abstained:
                return FailureDiagnosis(
                    primary=FailureType.FAILURE_TO_ABSTAIN,
                    evidence={**evidence, "annotated_answerable": False},
                    explanation="Question is annotated as unanswerable, but system attempted to generate an answer.",
                    confidence=1.0,
                )
            else:
                return FailureDiagnosis(
                    primary=FailureType.NO_FAILURE,
                    evidence={**evidence, "annotated_answerable": False},
                    explanation="Question is unanswerable and system correctly abstained.",
                    confidence=1.0,
                )

        # Rule 2: Insufficient ground truth / relevance labels
        if not relevant_set and not expected_answer:
            return FailureDiagnosis(
                primary=FailureType.UNKNOWN,
                evidence=evidence,
                explanation="Insufficient ground truth / relevance labels for diagnosis.",
                unclassified=True,
            )

        # Rule 3: Check retrieval quality
        hit_ids = retrieved_set.intersection(relevant_set) if relevant_set else set()
        evidence["hit_ids"] = list(hit_ids)

        if relevant_set and not hit_ids:
            if len(retrieved_docs) == 0:
                secondary = [FailureType.FAILURE_TO_ABSTAIN] if (has_answer and not is_abstained) else []
                return FailureDiagnosis(
                    primary=FailureType.MISSING_RELEVANT_EVIDENCE,
                    secondary=secondary,
                    evidence=evidence,
                    explanation="No documents retrieved while relevant evidence exists in corpus.",
                    confidence=0.95,
                )
            else:
                secondary = [FailureType.FAILURE_TO_ABSTAIN] if (has_answer and not is_abstained) else []
                return FailureDiagnosis(
                    primary=FailureType.WRONG_OR_IRRELEVANT_RETRIEVAL,
                    secondary=secondary,
                    evidence=evidence,
                    explanation="Retrieved documents do not contain annotated relevant evidence.",
                    confidence=0.95,
                )

        # Relevant evidence was retrieved (or relevance not restricted).
        # Rule 4: Incorrect abstention when sufficient evidence was retrieved
        if is_abstained:
            return FailureDiagnosis(
                primary=FailureType.INCORRECT_ABSTENTION,
                evidence=evidence,
                explanation="Relevant evidence was successfully retrieved, but the system abstained from answering.",
                confidence=0.9,
            )

        # Rule 5: Stale source check (requires metadata indication)
        stale_flag = ctx.get("is_stale")
        if stale_flag is None and judgment is not None:
            stale_flag = getattr(judgment, "metadata", {}).get("is_stale")
        if stale_flag:
            return FailureDiagnosis(
                primary=FailureType.STALE_SOURCE,
                evidence={**evidence, "stale_source_detected": True},
                explanation="Retrieved context originated from an outdated or stale document version.",
                confidence=0.85,
            )

        # Rule 6: Conflicting evidence check (requires metadata indication)
        conflict_flag = ctx.get("has_conflicts")
        if conflict_flag is None and judgment is not None:
            conflict_flag = getattr(judgment, "metadata", {}).get("conflicting_evidence")
        if conflict_flag:
            return FailureDiagnosis(
                primary=FailureType.CONFLICTING_EVIDENCE_MISHANDLED,
                evidence={**evidence, "conflicting_evidence_present": True},
                explanation="Contradictory evidence was retrieved and generator failed to resolve conflicting sources.",
                confidence=0.85,
            )

        # Rule 7: Context grounding and generation correctness
        ans_tokens = _tokenize(answer or "")
        ctx_tokens = _tokenize(full_context_text)
        exp_tokens = _tokenize(expected_answer or "") if expected_answer else set()

        ans_content = ans_tokens - _STOP_WORDS
        ctx_content = ctx_tokens - _STOP_WORDS
        if ans_content:
            faithfulness = len(ans_content & ctx_content) / len(ans_content)
        else:
            faithfulness = (len(ans_tokens & ctx_tokens) / len(ans_tokens)) if ans_tokens else 1.0
        evidence["faithfulness_token_ratio"] = round(faithfulness, 4)

        # Check if context was ignored (relevant evidence in retrieved docs, but answer has near-zero overlap with supplied context)
        if supplied_context is not None and len(ans_tokens) > 5:
            # Check overlap specifically with supplied context
            sup_tokens = _tokenize(supplied_context)
            if len(ans_tokens & sup_tokens) == 0 and len(sup_tokens) > 0:
                return FailureDiagnosis(
                    primary=FailureType.IGNORED_CONTEXT,
                    evidence={**evidence, "supplied_context_overlap": 0},
                    explanation="Relevant evidence was supplied to generator, but generation completely ignored it.",
                    confidence=0.85,
                )

        # Check correctness against expected answer
        is_correct = False
        if expected_answer:
            exact = answer.strip().lower() == expected_answer.strip().lower()
            overlap_exp = len(ans_tokens & exp_tokens)
            token_f1 = (2 * overlap_exp / (len(ans_tokens) + len(exp_tokens))) if (len(ans_tokens) + len(exp_tokens)) > 0 else 0.0
            evidence["token_f1"] = round(token_f1, 4)
            evidence["exact_match"] = exact

            if exact or token_f1 >= 0.7 or (exp_tokens and exp_tokens.issubset(ans_tokens)):
                is_correct = True

        if is_correct:
            return FailureDiagnosis(
                primary=FailureType.NO_FAILURE,
                evidence=evidence,
                explanation="Relevant evidence retrieved and answer matches expected reference.",
                confidence=0.95,
            )

        # Hallucination check: low context overlap on a substantive answer
        if len(ans_tokens) > 8 and faithfulness < 0.15:
            return FailureDiagnosis(
                primary=FailureType.UNSUPPORTED_HALLUCINATION,
                evidence=evidence,
                explanation="Answer contains substantive claims unsupported by the retrieved context.",
                confidence=0.8,
            )

        # If expected answer was provided and not matched
        if expected_answer and not is_correct:
            return FailureDiagnosis(
                primary=FailureType.INCOMPLETE_OR_INCORRECT_GENERATION,
                evidence=evidence,
                explanation="Relevant evidence was retrieved, but generation produced an incorrect or incomplete answer.",
                confidence=0.85,
            )

        # If no expected answer, but context overlap is solid and relevant docs were retrieved
        if faithfulness >= 0.5:
            return FailureDiagnosis(
                primary=FailureType.NO_FAILURE,
                evidence=evidence,
                explanation="Relevant evidence retrieved and answer is grounded in context.",
                confidence=0.75,
            )

        return FailureDiagnosis(
            primary=FailureType.UNKNOWN,
            evidence=evidence,
            explanation="Evidence is ambiguous or insufficient to assign a conclusive failure type.",
            unclassified=True,
        )
