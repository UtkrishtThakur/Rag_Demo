from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

from .schema import AnnotationRecord


def cohens_kappa(rater1: Sequence[str], rater2: Sequence[str]) -> float:
    """Calculate Cohen's Kappa between two parallel categorical ratings."""
    if len(rater1) != len(rater2):
        raise ValueError("Raters must have same number of items")
    n = len(rater1)
    if n == 0:
        return 0.0

    categories = sorted(set(list(rater1) + list(rater2)))
    agree = sum(1 for a, b in zip(rater1, rater2) if a == b)
    po = agree / n

    count1 = Counter(rater1)
    count2 = Counter(rater2)
    pe = sum((count1[c] / n) * (count2[c] / n) for c in categories)

    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0

    kappa = (po - pe) / (1.0 - pe)
    return float(round(kappa, 6))


def agreement_report(
    rater1: Sequence[str],
    rater2: Sequence[str],
    categories: Sequence[str] | None = None,
) -> dict[str, Any]:
    if len(rater1) != len(rater2):
        raise ValueError("Raters must have same number of items")
    n = len(rater1)
    agree = sum(1 for a, b in zip(rater1, rater2) if a == b)
    kappa = cohens_kappa(rater1, rater2)

    categories = list(categories) if categories is not None else sorted(set(list(rater1) + list(rater2)))
    cat_counts_1 = dict(Counter(rater1))
    cat_counts_2 = dict(Counter(rater2))

    return {
        "n": n,
        "raw_agreement": round(agree / n, 4) if n > 0 else 0.0,
        "cohens_kappa": kappa,
        "agreements": agree,
        "disagreements": n - agree,
        "categories": categories,
        "rater1_distribution": cat_counts_1,
        "rater2_distribution": cat_counts_2,
    }


def inter_rater_agreement(
    annotations: Sequence[AnnotationRecord],
    rater_a: str,
    rater_b: str,
    attribute: str = "primary_failure_type",
) -> dict[str, Any]:
    """Calculate agreement between two raters after strictly aligning by query_id."""
    records_a: dict[str, str] = {}
    records_b: dict[str, str] = {}
    duplicates_a: list[str] = []
    duplicates_b: list[str] = []

    for ann in annotations:
        val = getattr(ann, attribute, None)
        if val is None:
            continue
        val_str = str(val)
        if ann.rater_id == rater_a:
            if ann.query_id in records_a:
                duplicates_a.append(ann.query_id)
            records_a[ann.query_id] = val_str
        elif ann.rater_id == rater_b:
            if ann.query_id in records_b:
                duplicates_b.append(ann.query_id)
            records_b[ann.query_id] = val_str

    common_qids = sorted(set(records_a.keys()) & set(records_b.keys()))
    missing_in_b = sorted(set(records_a.keys()) - set(records_b.keys()))
    missing_in_a = sorted(set(records_b.keys()) - set(records_a.keys()))

    paired_a = [records_a[qid] for qid in common_qids]
    paired_b = [records_b[qid] for qid in common_qids]

    rep = agreement_report(paired_a, paired_b)

    disagreements_list = []
    for qid in common_qids:
        if records_a[qid] != records_b[qid]:
            disagreements_list.append({
                "query_id": qid,
                f"{rater_a}": records_a[qid],
                f"{rater_b}": records_b[qid],
            })

    return {
        "rater_a": rater_a,
        "rater_b": rater_b,
        "attribute": attribute,
        "n_common": len(common_qids),
        "missing_in_rater_b": missing_in_b,
        "missing_in_rater_a": missing_in_a,
        "duplicates_a": duplicates_a,
        "duplicates_b": duplicates_b,
        "raw_agreement": rep["raw_agreement"],
        "cohens_kappa": rep["cohens_kappa"],
        "disagreements": disagreements_list,
        "distribution_a": rep["rater1_distribution"],
        "distribution_b": rep["rater2_distribution"],
    }
