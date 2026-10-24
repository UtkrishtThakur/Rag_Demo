import tempfile
from pathlib import Path

from rag_inspector import RAGInspector
from rag_inspector.evaluation import EvaluationDataset, RelevanceJudgment
from rag_inspector.experiments import ExperimentConfig, ExperimentRunner


def test_experiment_runner():
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(output_dir=tmpdir)
        config = ExperimentConfig(experiment_id="test_exp_01", dataset_id="ds_01")

        run = RAGInspector().start_run("What is RAG?")
        run.record_retrieval([{"id": "d1", "text": "RAG is retrieval augmented generation."}])
        run.record_generation("RAG is retrieval augmented generation.")
        ex = run.finish()

        ds = EvaluationDataset()
        ds.add_judgment(RelevanceJudgment(query="What is RAG?", relevant_doc_ids={"d1"}))

        res = runner.run_experiment(config, [ex], dataset=ds)
        assert Path(res["json"]).exists()
        assert Path(res["md"]).exists()
        assert res["result"]["experiment_id"] == "test_exp_01"
