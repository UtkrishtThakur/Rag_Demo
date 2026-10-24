from rag_inspector.evaluation.metrics import (
    average_precision_at_k,
    compute_latency_statistics,
    dcg_at_k,
    lcs_length,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rouge_l,
)


def test_recall_at_k_basic():
    retrieved = ["d1", "d2", "d3", "d4", "d5"]
    relevant = {"d2", "d4"}
    assert recall_at_k(retrieved, relevant, 5) == 1.0
    assert recall_at_k(retrieved, relevant, 1) == 0.0


def test_recall_at_k_deduplication():
    # Duplicate retrieved document ID must not count multiple hits or exceed 1.0
    retrieved = ["d1", "d1", "d1", "d2"]
    relevant = {"d1"}
    assert recall_at_k(retrieved, relevant, 4) == 1.0


def test_recall_at_k_no_relevant():
    retrieved = ["d1", "d2"]
    relevant = set()
    assert recall_at_k(retrieved, relevant, 5) == 0.0


def test_precision_at_k_standard_denominator():
    retrieved = ["d1", "d2"]
    relevant = {"d1"}
    # Standard Cranfield denominator divides by cutoff k=5
    assert precision_at_k(retrieved, relevant, 5, denominator="k") == 0.2
    # Optional denominator divides by retrieved length 2
    assert precision_at_k(retrieved, relevant, 5, denominator="retrieved") == 0.5


def test_mrr():
    retrieved = ["d1", "d2", "d3"]
    relevant = {"d3"}
    assert mean_reciprocal_rank(retrieved, relevant) == 1 / 3
    relevant = {"d1"}
    assert mean_reciprocal_rank(retrieved, relevant) == 1.0
    relevant = {"d4"}
    assert mean_reciprocal_rank(retrieved, relevant) == 0.0


def test_ndcg():
    retrieved = ["d1", "d2", "d3"]
    rels = {"d1": 3.0, "d2": 0.0, "d3": 1.0}
    val = ndcg_at_k(retrieved, rels, 3)
    assert 0.0 <= val <= 1.0


def test_rouge_l_longest_common_subsequence():
    # Inverted sentence order demonstrates true LCS (length 4 of 6) rather than bag-of-words
    cand = "The capital of Australia is Canberra."
    ref = "Canberra is the capital of Australia."
    res = rouge_l(cand, ref)
    assert round(res["f1"], 4) == 0.6667
    assert round(res["recall"], 4) == 0.6667
    assert round(res["precision"], 4) == 0.6667

    # High overlap in same order yields > 0.8
    cand2 = "The capital of Australia is Canberra."
    ref2 = "The capital of Australia is Canberra city."
    res2 = rouge_l(cand2, ref2)
    assert res2["f1"] > 0.8
    assert res2["precision"] == 1.0


def test_latency_summary():
    latencies = [10.0, 20.0, 30.0, None, -5.0, 100.0]
    stats = compute_latency_statistics(latencies)
    assert stats["count_valid"] == 4  # 10, 20, 30, 100
    assert stats["count_missing"] == 1
    assert stats["count_invalid"] == 1
    assert stats["mean_ms"] == 40.0
    assert stats["median_ms"] == 25.0
    assert stats["min_ms"] == 10.0
    assert stats["max_ms"] == 100.0
