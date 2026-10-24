from rag_inspector import RAGInspector
from rag_inspector.evaluation import EvaluationConfig, EvaluationDataset, RAGEvaluator, RelevanceJudgment


def test_evaluate_single_execution():
    run = RAGInspector().start_run("What is RAG?")
    run.record_retrieval([
        {"id": "d1", "text": "RAG combines retrieval with generation.", "metadata": {"source": "docs.pdf"}},
        {"id": "d2", "text": "Other info."},
    ], latency_ms=12.5)
    run.record_generation("RAG combines retrieval and generation.", latency_ms=45.2)
    ex = run.finish()

    ds = EvaluationDataset()
    ds.add_judgment(RelevanceJudgment(
        query="What is RAG?",
        relevant_doc_ids={"d1"},
        relevance_scores={"d1": 3.0, "d2": 0.0},
        expected_answer="RAG combines retrieval and generation.",
    ))
    evaluator = RAGEvaluator(EvaluationConfig(k_values=[1, 5, 10]))
    report = evaluator.evaluate_single(ex, ds)

    assert report.aggregate.num_evaluated == 1
    assert report.aggregate.retrieval["recall@5"] == 1.0
    assert report.aggregate.retrieval["mrr"] == 1.0
    assert report.aggregate.retrieval["ndcg@1"] == 1.0
    assert report.aggregate.answer["exact_match"] == 1.0
    assert report.per_query[0].answer_metrics["verdict"] == "correct"
    assert report.aggregate.latency["retrieval"]["mean_ms"] == 12.5


def test_evaluate_multiple_executions():
    exs = []
    for q in ["Q1", "Q2"]:
        run = RAGInspector().start_run(q)
        if q == "Q1":
            run.record_retrieval([{"id": "a", "text": "passage a"}, {"id": "b", "text": "passage b"}])
        else:
            run.record_retrieval([{"id": "b", "text": "passage b"}, {"id": "c", "text": "passage c"}])
        run.record_generation(f"Answer to {q}")
        exs.append(run.finish())

    ds = EvaluationDataset()
    ds.add_judgment(RelevanceJudgment(query="Q1", relevant_doc_ids={"a"}))
    ds.add_judgment(RelevanceJudgment(query="Q2", relevant_doc_ids={"c"}))
    evaluator = RAGEvaluator()
    report = evaluator.evaluate_executions(exs, ds)
    assert report.aggregate.num_queries == 2
    assert report.aggregate.num_evaluated == 2


def test_empty_retrieval_handling():
    run = RAGInspector().start_run("Q")
    run.record_retrieval([])
    run.record_generation("I don't know.")
    ex = run.finish()
    ds = EvaluationDataset()
    ds.add_judgment(RelevanceJudgment(query="Q", relevant_doc_ids={"d1"}))
    evaluator = RAGEvaluator()
    report = evaluator.evaluate_single(ex, ds)
    assert report.per_query[0].retrieval_metrics["recall@5"] == 0.0
    assert report.per_query[0].abstention["is_abstention"] is True


def test_no_relevant_docs():
    run = RAGInspector().start_run("Q")
    run.record_retrieval([{"id": "d1", "text": "passage content"}])
    run.record_generation("ans")
    ex = run.finish()
    ds = EvaluationDataset()
    ds.add_judgment(RelevanceJudgment(query="Q", relevant_doc_ids=set()))
    evaluator = RAGEvaluator()
    report = evaluator.evaluate_single(ex, ds)
    assert report.aggregate.num_evaluated == 1


def test_faithfulness():
    run = RAGInspector().start_run("Q")
    run.record_retrieval([{"text": "The sky is blue."}])
    run.record_generation("The sky is blue.")
    ex = run.finish()
    evaluator = RAGEvaluator()
    report = evaluator.evaluate_single(ex)
    assert report.per_query[0].faithfulness_metrics["faithfulness"] == 1.0
