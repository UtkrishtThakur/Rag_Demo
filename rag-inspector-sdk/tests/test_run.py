import pytest

from rag_inspector import (
    ExecutionRecord,
    InvalidAnswerError,
    InvalidDocumentError,
    InvalidRetrievalError,
    MissingPipelineDataError,
    RAGInspector,
    RunStateError,
)


def new_run():
    return RAGInspector().start_run("What is RAG?")


def test_complete_run():
    run = new_run()
    run.record_retrieval([
        {"id": "d1", "text": "RAG retrieves context.", "metadata": {"page": 1}},
        "Another passage.",
    ], latency_ms=15.0)
    run.record_generation("RAG combines retrieval and generation.", context="RAG retrieves context.", latency_ms=30.0)
    result = run.finish()

    assert result.status.value == "completed"
    assert len(result.retrieved_documents) == 2
    assert result.retrieved_documents[0].id == "d1"
    assert result.retrieved_documents[1].text == "Another passage."
    assert result.answer == "RAG combines retrieval and generation."
    assert result.supplied_context == "RAG retrieves context."
    assert result.retrieval_latency_ms == 15.0
    assert result.generation_latency_ms == 30.0
    assert result.finished_at


def test_empty_retrieval_is_valid_recorded_data():
    run = new_run().record_retrieval([])
    run.record_generation("I don't know.")
    assert run.finish().retrieved_documents == []


def test_finish_reports_missing_stages():
    run = new_run()
    with pytest.raises(MissingPipelineDataError) as exc:
        run.finish()
    assert "retrieval documents" in str(exc.value)
    assert "generated answer" in str(exc.value)


def test_finish_reports_missing_generation_after_retrieval():
    run = new_run().record_retrieval(["source text"])
    with pytest.raises(MissingPipelineDataError, match="generated answer"):
        run.finish()


def test_bad_retrieval_payload():
    with pytest.raises(InvalidRetrievalError):
        new_run().record_retrieval("not a list of documents")


def test_bad_document():
    with pytest.raises(InvalidDocumentError, match="non-empty string field 'text'"):
        new_run().record_retrieval([{"id": "d1"}])


def test_framework_object_is_not_silently_stringified():
    class FrameworkDocument:
        page_content = "text"

    with pytest.raises(InvalidDocumentError, match="Convert framework objects explicitly"):
        new_run().record_retrieval([FrameworkDocument()])


@pytest.mark.parametrize("answer", ["", " ", None, 123])
def test_invalid_answer(answer):
    with pytest.raises(InvalidAnswerError):
        new_run().record_generation(answer)


def test_completed_run_is_immutable():
    run = new_run()
    run.record_retrieval([])
    run.record_generation("answer")
    run.finish()
    with pytest.raises(RunStateError):
        run.record_generation("another answer")


def test_to_dict_keeps_partial_run_data():
    run = new_run()
    run.record_retrieval(["source"])
    record = run.to_dict()
    assert record["status"] == "recording"
    assert record["retrieved_documents"][0]["text"] == "source"
    assert record["answer"] is None


def test_record_failure_preserves_telemetry():
    run = new_run()
    run.record_retrieval(["document text"])
    failed = run.record_failure("LLM generation timeout")
    assert failed.status.value == "failed"
    assert failed.error == "LLM generation timeout"
    assert len(failed.retrieved_documents) == 1
    assert failed.finished_at is not None


def test_execution_record_roundtrip_serialization():
    run = new_run()
    run.record_retrieval([{"id": "doc1", "text": "passage", "score": 0.85, "rank": 1}])
    run.record_generation("generated answer", context="supplied context")
    record = run.finish()

    serialized = record.to_dict()
    restored = ExecutionRecord.from_dict(serialized)

    assert restored.run_id == record.run_id
    assert restored.question == record.question
    assert restored.supplied_context == "supplied context"
    assert restored.retrieved_documents[0].id == "doc1"
    assert restored.retrieved_documents[0].score == 0.85
    assert restored.answer == "generated answer"
