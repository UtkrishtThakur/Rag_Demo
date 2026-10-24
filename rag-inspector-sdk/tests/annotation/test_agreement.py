import pytest
from rag_inspector.annotation import (
    AnnotationRecord,
    AnnotationSet,
    agreement_report,
    cohens_kappa,
    inter_rater_agreement,
)


def test_cohens_kappa_perfect_agreement():
    r1 = ["correct", "incorrect", "partial"]
    r2 = ["correct", "incorrect", "partial"]
    assert cohens_kappa(r1, r2) == 1.0


def test_cohens_kappa_disagreement():
    r1 = ["correct", "incorrect", "correct", "partial"]
    r2 = ["correct", "incorrect", "correct", "correct"]
    k = cohens_kappa(r1, r2)
    assert -1.0 <= k <= 1.0
    rep = agreement_report(r1, r2)
    assert rep["n"] == 4
    assert rep["raw_agreement"] == 0.75


def test_inter_rater_agreement_aligned_by_query_id():
    # Rater A and Rater B annotations provided in shuffled order
    anns = [
        AnnotationRecord(query_id="q2", query="What is B?", rater_id="rater_b", primary_failure_type="no_failure"),
        AnnotationRecord(query_id="q1", query="What is A?", rater_id="rater_a", primary_failure_type="wrong_or_irrelevant_retrieval"),
        AnnotationRecord(query_id="q1", query="What is A?", rater_id="rater_b", primary_failure_type="wrong_or_irrelevant_retrieval"),
        AnnotationRecord(query_id="q2", query="What is B?", rater_id="rater_a", primary_failure_type="no_failure"),
        AnnotationRecord(query_id="q3", query="What is C?", rater_id="rater_a", primary_failure_type="unsupported_hallucination"),
    ]

    res = inter_rater_agreement(anns, rater_a="rater_a", rater_b="rater_b")
    assert res["n_common"] == 2
    assert res["raw_agreement"] == 1.0
    assert "q3" in res["missing_in_rater_b"]


def test_annotation_record_validation():
    with pytest.raises(ValueError, match="query_id"):
        AnnotationRecord(query_id="", query="Q", rater_id="r1")

    with pytest.raises(ValueError, match="answer_correctness"):
        AnnotationRecord(query_id="q1", query="Q", rater_id="r1", answer_correctness="invalid_label")
