from rag_inspector.judge import (
    JudgeResult,
    JudgeStabilityExperiment,
    RuleBasedJudgeProvider,
)


def test_rule_based_judge():
    judge = RuleBasedJudgeProvider()
    res = judge.judge(
        query="What is RAG?",
        answer="RAG retrieves documents and generates answers.",
        context=["RAG is a retrieval augmented generation system."],
        query_id="q1",
    )
    assert res.query_id == "q1"
    assert res.verdict == "correct"


def test_compare_judges_agreement():
    exp = JudgeStabilityExperiment(seed=42)
    j1_results = [
        JudgeResult(query_id="q1", verdict="correct", score=1.0),
        JudgeResult(query_id="q2", verdict="incorrect", score=0.0),
    ]
    j2_results = [
        JudgeResult(query_id="q1", verdict="correct", score=0.9),
        JudgeResult(query_id="q2", verdict="incorrect", score=0.1),
    ]

    report = exp.compare_judges(j1_results, j2_results, "j1", "j2")
    assert report.agreement == 1.0
    assert report.cohens_kappa == 1.0
    assert report.verdict_changes == 0
    assert report.score_drift_mean_abs == 0.1


def test_pairwise_order_bias():
    exp = JudgeStabilityExperiment(seed=42)
    judge = RuleBasedJudgeProvider()

    # Define pairwise test cases
    pairs = [
        {
            "query_id": "p1",
            "query": "What is Python?",
            "answer_a": "Python is a programming language.",
            "answer_b": "Python is a snake species.",
            "context": ["Python is a high-level programming language created by Guido van Rossum."],
        }
    ]

    def pairwise_fn(q, a1, a2, ctx):
        res = judge.judge_pairwise(q, a1, a2, ctx)
        return res.verdict

    order_report = exp.test_pairwise_order_bias(pairwise_fn, pairs)
    assert order_report["total_pairs"] == 1
    assert "inconsistency_rate" in order_report
    assert "order_bias_detected" in order_report
