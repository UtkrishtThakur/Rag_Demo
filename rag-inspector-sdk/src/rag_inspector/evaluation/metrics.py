from __future__ import annotations

import math
import re
from typing import Any, Sequence, Set


def _get_k(k: int) -> int:
    if k <= 0:
        raise ValueError("k must be > 0")
    return k


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str] | Sequence[str],
    k: int,
) -> float:
    """Recall@k: fraction of unique relevant documents retrieved in top-k."""
    k = _get_k(k)
    relevant_set = {str(r).strip() for r in relevant_ids if str(r).strip()}
    if not relevant_set:
        return 0.0

    retrieved_at_k = [str(r).strip() for r in retrieved_ids[:k] if str(r).strip()]
    seen_hits: set[str] = set()
    for rid in retrieved_at_k:
        if rid in relevant_set:
            seen_hits.add(rid)

    hits = len(seen_hits)
    return min(1.0, hits / len(relevant_set))


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str] | Sequence[str],
    k: int,
    denominator: str = "k",
) -> float:
    """Precision@k: unique relevant documents in top-k divided by cutoff k (or retrieved length)."""
    k = _get_k(k)
    relevant_set = {str(r).strip() for r in relevant_ids if str(r).strip()}
    retrieved_at_k = [str(r).strip() for r in retrieved_ids[:k] if str(r).strip()]
    if not retrieved_at_k:
        return 0.0

    seen_hits: set[str] = set()
    for rid in retrieved_at_k:
        if rid in relevant_set:
            seen_hits.add(rid)

    hits = len(seen_hits)
    denom = k if denominator == "k" else len(retrieved_at_k)
    if denom <= 0:
        return 0.0
    return hits / denom


def mean_reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str] | Sequence[str],
    k: int | None = None,
) -> float:
    """Reciprocal rank of first relevant document."""
    relevant_set = {str(r).strip() for r in relevant_ids if str(r).strip()}
    if not relevant_set:
        return 0.0
    retrieved = retrieved_ids if k is None else retrieved_ids[:_get_k(k)]
    for i, rid in enumerate(retrieved):
        clean_id = str(rid).strip()
        if clean_id and clean_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def average_precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str] | Sequence[str],
    k: int,
) -> float:
    """Average Precision at rank cutoff k with deduplicated hit counting."""
    k = _get_k(k)
    relevant_set = {str(r).strip() for r in relevant_ids if str(r).strip()}
    if not relevant_set:
        return 0.0

    retrieved_at_k = retrieved_ids[:k]
    score = 0.0
    seen_hits: set[str] = set()

    for i, rid in enumerate(retrieved_at_k):
        clean_id = str(rid).strip()
        if clean_id and clean_id in relevant_set and clean_id not in seen_hits:
            seen_hits.add(clean_id)
            score += len(seen_hits) / (i + 1)

    return score / len(relevant_set)


def mean_average_precision(
    retrieved_lists: Sequence[Sequence[str]],
    relevant_lists: Sequence[Set[str] | Sequence[str]],
) -> float:
    if len(retrieved_lists) != len(relevant_lists):
        raise ValueError("retrieved_lists and relevant_lists must have same length")
    if not retrieved_lists:
        return 0.0
    aps = []
    for retrieved_ids, relevant_ids in zip(retrieved_lists, relevant_lists):
        relevant_set = {str(r).strip() for r in relevant_ids if str(r).strip()}
        if not relevant_set:
            continue
        ap = average_precision_at_k(retrieved_ids, relevant_set, len(retrieved_ids))
        aps.append(ap)
    if not aps:
        return 0.0
    return sum(aps) / len(aps)


def dcg_at_k(retrieved_ids: Sequence[str], relevances: dict[str, float], k: int) -> float:
    k = _get_k(k)
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        clean_id = str(rid).strip()
        rel = relevances.get(clean_id, 0.0)
        dcg += (2 ** rel - 1) / math.log2(i + 2)
    return dcg


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevances: dict[str, float],
    k: int,
) -> float:
    k = _get_k(k)
    dcg = dcg_at_k(retrieved_ids, relevances, k)
    ideal_rels = sorted(relevances.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        idcg += (2 ** rel - 1) / math.log2(i + 2)
    if idcg == 0.0:
        return 0.0
    return min(1.0, dcg / idcg)


def _tokenize_list(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def lcs_length(seq1: list[str], seq2: list[str]) -> int:
    """Compute the Longest Common Subsequence length between two token sequences."""
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return 0
    # Use single-row DP table for memory efficiency
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def rouge_l(candidate: str, reference: str) -> dict[str, float]:
    """Compute standard ROUGE-L using Longest Common Subsequence."""
    cand_tokens = _tokenize_list(candidate)
    ref_tokens = _tokenize_list(reference)

    if not cand_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = lcs_length(cand_tokens, ref_tokens)
    prec = lcs / len(cand_tokens)
    rec = lcs / len(ref_tokens)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }


def compute_latency_statistics(latencies: Sequence[float | None]) -> dict[str, Any]:
    """Compute mean, median, p95 and observation counts for valid latencies in milliseconds."""
    valid: list[float] = []
    missing_count = 0
    invalid_count = 0

    for x in latencies:
        if x is None:
            missing_count += 1
        elif isinstance(x, (int, float)) and x >= 0:
            valid.append(float(x))
        else:
            invalid_count += 1

    if not valid:
        return {
            "count_valid": 0,
            "count_missing": missing_count,
            "count_invalid": invalid_count,
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
        }

    valid.sort()
    n = len(valid)
    mean_val = sum(valid) / n
    median_val = valid[n // 2] if n % 2 != 0 else (valid[n // 2 - 1] + valid[n // 2]) / 2.0

    p95_idx = int(math.ceil(0.95 * n)) - 1
    p95_val = valid[max(0, min(n - 1, p95_idx))]

    return {
        "count_valid": n,
        "count_missing": missing_count,
        "count_invalid": invalid_count,
        "mean_ms": round(mean_val, 2),
        "median_ms": round(median_val, 2),
        "p95_ms": round(p95_val, 2),
        "min_ms": round(valid[0], 2),
        "max_ms": round(valid[-1], 2),
    }
