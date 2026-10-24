import pytest

from rag_inspector import InvalidQuestionError, RAGInspector


def test_start_run_creates_recording_run():
    run = RAGInspector().start_run("What is RAG?")
    assert run.run_id
    assert run.status.value == "recording"
    assert run.to_dict()["question"] == "What is RAG?"


@pytest.mark.parametrize("question", ["", "   ", None, 123])
def test_invalid_question_raises(question):
    with pytest.raises(InvalidQuestionError):
        RAGInspector().start_run(question)
