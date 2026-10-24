from rag_inspector import RAGInspector
from rag_inspector.benchmark import BenchmarkRunner


def test_benchmark_runner_evaluation_only():
    exs = []
    for i in range(5):
        run = RAGInspector().start_run(f"Q{i}", query_id=f"q_{i}")
        run.record_retrieval([{"id": "d1", "text": "passage content"}], latency_ms=10.0)
        run.record_generation("answer", latency_ms=20.0)
        exs.append(run.finish())

    runner = BenchmarkRunner()
    res = runner.run_benchmark(exs, target_minutes=1.0)
    assert res.benchmark_mode == "evaluation_only"
    assert res.num_questions == 5
    assert res.duration_seconds >= 0.0
    assert res.met_target is True
    assert res.success_count == 5


def test_benchmark_runner_end_to_end():
    def mock_pipeline(question: str):
        inspector = RAGInspector()
        run = inspector.start_run(question)
        run.record_retrieval([{"id": "chunk_1", "text": "content"}])
        run.record_generation("mock answer")
        ex = run.finish()
        return {"answer": "mock answer", "rag_inspector": ex.to_dict()}

    runner = BenchmarkRunner()
    questions = ["What is A?", "What is B?"]
    res = runner.run_end_to_end_benchmark(mock_pipeline, questions, target_minutes=1.0)
    assert res.benchmark_mode == "end_to_end"
    assert res.num_questions == 2
    assert res.success_count == 2
    assert res.met_target is True
