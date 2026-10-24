from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass
class JudgeResult:
    query_id: str
    verdict: str  # e.g., "A", "B", "tie", "correct", "incorrect"
    score: float | None = None
    explanation: str = ""
    confidence: float | None = None
    provider: str = ""
    model: str = ""
    prompt_version: str = "v1"
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JudgeResult":
        return cls(**data)


class JudgeProvider(Protocol):
    name: str
    model: str

    def judge(
        self,
        query: str,
        answer: str,
        context: list[str],
        query_id: str = "",
        **kwargs: Any,
    ) -> JudgeResult:
        ...

    def judge_pairwise(
        self,
        query: str,
        answer_a: str,
        answer_b: str,
        context: list[str],
        query_id: str = "",
        **kwargs: Any,
    ) -> JudgeResult:
        ...


class RuleBasedJudgeProvider:
    """Configurable, deterministic rule-based judge for testing and offline evaluation."""

    def __init__(self, name: str = "rule_judge", model: str = "heuristic_v1"):
        self.name = name
        self.model = model

    def judge(
        self,
        query: str,
        answer: str,
        context: list[str],
        query_id: str = "",
        expected_answer: str | None = None,
        **kwargs: Any,
    ) -> JudgeResult:
        ans_clean = answer.strip().lower()
        if not ans_clean:
            return JudgeResult(
                query_id=query_id,
                verdict="incorrect",
                score=0.0,
                explanation="Answer is empty.",
                provider=self.name,
                model=self.model,
            )

        if expected_answer:
            exp_clean = expected_answer.strip().lower()
            if ans_clean == exp_clean or exp_clean in ans_clean:
                return JudgeResult(
                    query_id=query_id,
                    verdict="correct",
                    score=1.0,
                    explanation="Answer matches reference.",
                    provider=self.name,
                    model=self.model,
                )

        # Context support check
        ctx_text = " ".join(context).lower()
        ctx_words = set(re.findall(r"\w+", ctx_text))
        words = [w for w in re.findall(r"\w+", ans_clean) if len(w) > 3]
        matched = sum(
            1 for w in words
            if w in ctx_words or any(cw.startswith(w[:4]) or w.startswith(cw[:4]) for cw in ctx_words if len(cw) >= 4)
        )
        ratio = (matched / len(words)) if words else 1.0

        if ratio >= 0.5:
            return JudgeResult(
                query_id=query_id,
                verdict="correct",
                score=round(ratio, 2),
                explanation="Answer grounded in context.",
                provider=self.name,
                model=self.model,
            )
        return JudgeResult(
            query_id=query_id,
            verdict="incorrect",
            score=round(ratio, 2),
            explanation="Insufficient grounding in context.",
            provider=self.name,
            model=self.model,
        )

    def judge_pairwise(
        self,
        query: str,
        answer_a: str,
        answer_b: str,
        context: list[str],
        query_id: str = "",
        **kwargs: Any,
    ) -> JudgeResult:
        res_a = self.judge(query, answer_a, context, query_id=query_id)
        res_b = self.judge(query, answer_b, context, query_id=query_id)

        score_a = res_a.score or 0.0
        score_b = res_b.score or 0.0

        if score_a > score_b:
            verdict = "A"
        elif score_b > score_a:
            verdict = "B"
        else:
            verdict = "tie"

        return JudgeResult(
            query_id=query_id,
            verdict=verdict,
            score=abs(score_a - score_b),
            explanation=f"A={score_a}, B={score_b}",
            provider=self.name,
            model=self.model,
        )
