from rag_inspector import RAGInspector
from rag_inspector.comparison import compare_versions


def test_compare_versions_basic():
    exs1 = []
    exs2 = []
    for i in range(3):
        r1 = RAGInspector().start_run(f"Q{i}", query_id=f"q_{i}")
        r1.record_retrieval([], latency_ms=10.0)
        r1.record_generation("a", latency_ms=20.0)
        exs1.append(r1.finish())

        r2 = RAGInspector().start_run(f"Q{i}", query_id=f"q_{i}")
        r2.record_retrieval([{"id": "d1", "text": "passage content"}], latency_ms=15.0)
        r2.record_generation("a", latency_ms=25.0)
        exs2.append(r2.finish())

    comp = compare_versions(exs1, exs2, "v1", "v2")
    assert comp.version_a == "v1"
    assert comp.version_b == "v2"
    assert comp.paired is True
    assert len(comp.per_query) == 3
    assert len(comp.missing_matches) == 0
    assert "num_retrieved" in comp.deltas
    assert comp.deltas["num_retrieved"] == 1.0


def test_compare_versions_unmatched_and_zero_baseline():
    r1 = RAGInspector().start_run("Q1", query_id="q1")
    r1.record_retrieval([])
    r1.record_generation("ans1")
    ex1 = r1.finish()

    r2 = RAGInspector().start_run("Q2", query_id="q2")
    r2.record_retrieval([])
    r2.record_generation("ans2")
    ex2 = r2.finish()

    comp = compare_versions([ex1], [ex2], "v1", "v2")
    assert len(comp.per_query) == 0
    assert "q1" in comp.missing_matches
    assert "q2" in comp.missing_matches


def test_compare_versions_relative_change_zero_baseline():
    r1 = RAGInspector().start_run("Q1", query_id="q1")
    r1.record_retrieval([])  # num_retrieved = 0
    r1.record_generation("ans1")
    ex1 = r1.finish()

    r2 = RAGInspector().start_run("Q1", query_id="q1")
    r2.record_retrieval([{"id": "d1", "text": "text"}])  # num_retrieved = 1
    r2.record_generation("ans1")
    ex2 = r2.finish()

    comp = compare_versions([ex1], [ex2], "v1", "v2")
    assert comp.metrics_a["num_retrieved"] == 0.0
    assert comp.metrics_b["num_retrieved"] == 1.0
    # Relative change from 0 is mathematically undefined; represented as None
    assert comp.relative_changes["num_retrieved"] is None
