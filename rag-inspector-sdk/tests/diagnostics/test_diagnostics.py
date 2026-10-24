from rag_inspector import RAGInspector
from rag_inspector.diagnostics import (
    FailureClassifier,
    FailureType,
    classification_report,
    confusion_matrix,
)
from rag_inspector.evaluation import EvaluationDataset, RelevanceJudgment


def test_diagnosis_no_failure_on_correct_answer():
    run = RAGInspector().start_run("What is the capital of France?")
    run.record_retrieval([{"id": "d1", "text": "Paris is the capital of France."}])
    run.record_generation("Paris is the capital of France.")
    ex = run.finish()

    ds = EvaluationDataset()
    ds.add_judgment(RelevanceJudgment(
        query="What is the capital of France?",
        relevant_doc_ids={"d1"},
        expected_answer="Paris is the capital of France.",
    ))
    clf = FailureClassifier()
    diag = clf.classify(ex, ds)
    assert diag.primary == FailureType.NO_FAILURE
    assert diag.unclassified is False


def test_diagnosis_missing_relevant_evidence():
    run = RAGInspector().start_run("Q")
    run.record_retrieval([])
    run.record_generation("I don't know.")
    ex = run.finish()

    ds = EvaluationDataset()
    ds.add_judgment(RelevanceJudgment(query="Q", relevant_doc_ids={"d1"}))
    clf = FailureClassifier()
    diag = clf.classify(ex, ds)
    assert diag.primary == FailureType.MISSING_RELEVANT_EVIDENCE


def test_diagnosis_wrong_or_irrelevant_retrieval():
    run = RAGInspector().start_run("Q")
    run.record_retrieval([{"id": "d99", "text": "Completely irrelevant content."}])
    run.record_generation("Some answer.")
    ex = run.finish()

    ds = EvaluationDataset()
    ds.add_judgment(RelevanceJudgment(query="Q", relevant_doc_ids={"d1"}))
    clf = FailureClassifier()
    diag = clf.classify(ex, ds)
    assert diag.primary == FailureType.WRONG_OR_IRRELEVANT_RETRIEVAL


def test_diagnosis_unsupported_hallucination():
    run = RAGInspector().start_run("Q")
    run.record_retrieval([{"id": "d1", "text": "Canberra is the capital city of Australia."}])
    run.record_generation("The president of Antarctica announced a new colony on Saturn with rocket engines.")
    ex = run.finish()

    ds = EvaluationDataset()
    ds.add_judgment(RelevanceJudgment(query="Q", relevant_doc_ids={"d1"}, expected_answer="Canberra is the capital."))
    clf = FailureClassifier()
    diag = clf.classify(ex, ds)
    assert diag.primary == FailureType.UNSUPPORTED_HALLUCINATION


def test_diagnosis_incorrect_abstention():
    run = RAGInspector().start_run("Q")
    run.record_retrieval([{"id": "d1", "text": "Canberra is the capital city of Australia."}])
    run.record_generation("I am unable to answer based on the provided context.")
    ex = run.finish()

    ds = EvaluationDataset()
    ds.add_judgment(RelevanceJudgment(query="Q", relevant_doc_ids={"d1"}))
    clf = FailureClassifier()
    diag = clf.classify(ex, ds)
    assert diag.primary == FailureType.INCORRECT_ABSTENTION


def test_diagnosis_failure_to_abstain():
    run = RAGInspector().start_run("Q")
    run.record_retrieval([{"id": "d1", "text": "General text."}])
    run.record_generation("Confident invented answer.")
    ex = run.finish()

    ds = EvaluationDataset()
    ds.add_judgment(RelevanceJudgment(query="Q", metadata={"answerable": False}))
    clf = FailureClassifier()
    diag = clf.classify(ex, ds)
    assert diag.primary == FailureType.FAILURE_TO_ABSTAIN


def test_confusion_matrix_and_report():
    y_true = ["no_failure", "wrong_or_irrelevant_retrieval", "no_failure"]
    y_pred = ["no_failure", "wrong_or_irrelevant_retrieval", "unsupported_hallucination"]

    cm = confusion_matrix(y_true, y_pred)
    assert "labels" in cm
    assert "matrix" in cm

    rep = classification_report(y_true, y_pred)
    assert "macro_f1" in rep
    assert "no_failure" in rep["per_class"]
    assert rep["total_samples"] == 3
