from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..annotation.agreement import agreement_report, cohens_kappa
from .base import JudgeResult


@dataclass
class JudgeStabilityReport:
    judge_a: str
    judge_b: str
    agreement: float = 0.0
    cohens_kappa: float = 0.0
    verdict_changes: int = 0
    score_drift_mean_abs: float = 0.0
    order_bias_detected: bool = False
    order_bias_details: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "judge_a": self.judge_a,
            "judge_b": self.judge_b,
            "agreement": self.agreement,
            "cohens_kappa": self.cohens_kappa,
            "verdict_changes": self.verdict_changes,
            "score_drift_mean_abs": self.score_drift_mean_abs,
            "order_bias_detected": self.order_bias_detected,
            "order_bias_details": dict(self.order_bias_details),
            "config": dict(self.config),
            "results": list(self.results),
        }


class JudgeStabilityExperiment:
    """Evaluate inter-judge reliability and pairwise answer presentation order bias."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def compare_judges(
        self,
        results_a: Sequence[JudgeResult | dict[str, Any]],
        results_b: Sequence[JudgeResult | dict[str, Any]],
        judge_a: str = "judge1",
        judge_b: str = "judge2",
    ) -> JudgeStabilityReport:
        # Match strictly by query_id
        map_a: dict[str, Any] = {}
        for r in results_a:
            qid = getattr(r, "query_id", None) if not isinstance(r, dict) else r.get("query_id")
            if qid:
                map_a[str(qid)] = r

        map_b: dict[str, Any] = {}
        for r in results_b:
            qid = getattr(r, "query_id", None) if not isinstance(r, dict) else r.get("query_id")
            if qid:
                map_b[str(qid)] = r

        common_keys = sorted(set(map_a.keys()) & set(map_b.keys()))
        if not common_keys:
            return JudgeStabilityReport(judge_a=judge_a, judge_b=judge_b, config={"seed": self.seed})

        verdicts_a = []
        verdicts_b = []
        score_diffs = []
        detailed_rows = []

        for k in common_keys:
            item_a = map_a[k]
            item_b = map_b[k]

            va = getattr(item_a, "verdict", None) if not isinstance(item_a, dict) else item_a.get("verdict")
            vb = getattr(item_b, "verdict", None) if not isinstance(item_b, dict) else item_b.get("verdict")
            verdicts_a.append(str(va))
            verdicts_b.append(str(vb))

            sa = getattr(item_a, "score", None) if not isinstance(item_a, dict) else item_a.get("score")
            sb = getattr(item_b, "score", None) if not isinstance(item_b, dict) else item_b.get("score")

            if sa is not None and sb is not None:
                score_diffs.append(abs(float(sa) - float(sb)))

            detailed_rows.append({
                "query_id": k,
                "verdict_a": va,
                "verdict_b": vb,
                "agrees": str(va) == str(vb),
                "score_a": sa,
                "score_b": sb,
            })

        rep = agreement_report(verdicts_a, verdicts_b)
        mean_score_drift = (sum(score_diffs) / len(score_diffs)) if score_diffs else 0.0

        return JudgeStabilityReport(
            judge_a=judge_a,
            judge_b=judge_b,
            agreement=rep["raw_agreement"],
            cohens_kappa=rep["cohens_kappa"],
            verdict_changes=rep["disagreements"],
            score_drift_mean_abs=round(mean_score_drift, 4),
            config={"seed": self.seed, "paired_queries": len(common_keys)},
            results=detailed_rows,
        )

    def test_pairwise_order_bias(
        self,
        pairwise_judge_fn: Callable[[str, str, str, list[str]], str],
        pairs: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        """Test whether answer presentation order (A vs B) systematically biases the judge.

        Each pair dict should have: query, answer_a, answer_b, context, and optional query_id.
        pairwise_judge_fn(query, cand1, cand2, context) -> "A" (meaning cand1), "B" (cand2), or "tie".
        """
        if not pairs:
            return {"count": 0, "order_bias_detected": False}

        inconsistent_count = 0
        cand1_wins = 0
        cand2_wins = 0
        ties = 0

        details = []

        for i, pair in enumerate(pairs):
            q = pair["query"]
            ans_a = pair["answer_a"]
            ans_b = pair["answer_b"]
            ctx = pair.get("context", [])
            qid = pair.get("query_id", f"pair_{i}")

            # Forward order: (Candidate A, Candidate B)
            res_forward = str(pairwise_judge_fn(q, ans_a, ans_b, ctx)).strip().upper()

            # Reverse order: (Candidate B, Candidate A)
            res_reversed = str(pairwise_judge_fn(q, ans_b, ans_a, ctx)).strip().upper()

            # Translate: In forward order, "A" means Candidate A won, "B" means Candidate B won
            # In reversed order, "A" means Candidate B won, "B" means Candidate A won
            favored_forward = "A" if res_forward == "A" else ("B" if res_forward == "B" else "tie")
            favored_reversed = "B" if res_reversed == "A" else ("A" if res_reversed == "B" else "tie")

            is_consistent = (favored_forward == favored_reversed)
            if not is_consistent:
                inconsistent_count += 1

            if res_forward == "A":
                cand1_wins += 1
            elif res_forward == "B":
                cand2_wins += 1
            else:
                ties += 1

            details.append({
                "query_id": qid,
                "forward_verdict": res_forward,
                "reversed_verdict": res_reversed,
                "consistent": is_consistent,
            })

        n = len(pairs)
        inconsistency_rate = inconsistent_count / n
        pos1_rate = cand1_wins / n
        pos2_rate = cand2_wins / n

        # Detect order bias if inconsistency rate is high or strong position skew exists
        order_bias = (inconsistency_rate > 0.15) or (abs(pos1_rate - pos2_rate) > 0.25)

        return {
            "total_pairs": n,
            "inconsistent_pairs": inconsistent_count,
            "inconsistency_rate": round(inconsistency_rate, 4),
            "position_1_win_rate": round(pos1_rate, 4),
            "position_2_win_rate": round(pos2_rate, 4),
            "tie_rate": round(ties / n, 4),
            "order_bias_detected": order_bias,
            "details": details,
        }

    def shuffle_dataset(self, items: Sequence[Any], seed: int | None = None) -> list[Any]:
        """Randomize dataset item order reproducibly."""
        s = seed if seed is not None else self.seed
        rng = random.Random(s)
        items_copy = list(items)
        rng.shuffle(items_copy)
        return items_copy
