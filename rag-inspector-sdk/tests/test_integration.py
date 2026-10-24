"""Integration test verifying end-to-end data flow:
Chroma result -> normalized retrieval -> retrieved document view -> SDK record_retrieval()
-> ExecutionRecord -> evaluation dataset matching.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure repo root is on sys.path
_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from retrival.retrieval import RetrievalApp
from rag_inspector import (
    EvaluationConfig,
    EvaluationDataset,
    RAGEvaluator,
    RelevanceJudgment,
)
from retrival.vectordb import RetrievedChunk


def test_chroma_id_preservation_through_sdk():
    app = RetrievalApp()

    # Mock DB search to return specific Chroma chunk IDs
    sample_chunks = [
        RetrievedChunk(id="chroma_chunk_001", text="Canberra is the capital of Australia.", distance=0.12, metadata={"file_name": "aus.pdf", "page_number": 1}),
        RetrievedChunk(id="chroma_chunk_002", text="Sydney is the largest city in Australia.", distance=0.45, metadata={"file_name": "aus.pdf", "page_number": 2}),
    ]
    app.db.search = MagicMock(return_value=sample_chunks)
    app.embedder.embed_query = MagicMock(return_value=[0.1] * 384)
    app.llm.generate = MagicMock(return_value="The capital of Australia is Canberra.")

    result = app.answer_question("What is the capital of Australia?")

    # 1. Verify Chroma ID survived into chunks_data
    assert "chunks" in result
    assert len(result["chunks"]) == 2
    assert result["chunks"][0]["id"] == "chroma_chunk_001"
    assert result["chunks"][1]["id"] == "chroma_chunk_002"

    # 2. Verify Chroma ID survived into SDK ExecutionRecord
    assert "rag_inspector" in result
    exec_data = result["rag_inspector"]
    retrieved_docs = exec_data["retrieved_documents"]
    assert len(retrieved_docs) == 2
    assert retrieved_docs[0]["id"] == "chroma_chunk_001"
    assert retrieved_docs[1]["id"] == "chroma_chunk_002"

    # 3. Verify evaluation against ground truth using these preserved IDs
    from rag_inspector.schemas import ExecutionRecord
    execution = ExecutionRecord.from_dict(exec_data)

    dataset = EvaluationDataset(name="geo_test")
    dataset.add_judgment(RelevanceJudgment(
        query="What is the capital of Australia?",
        relevant_doc_ids={"chroma_chunk_001"},
        expected_answer="The capital of Australia is Canberra.",
    ))

    evaluator = RAGEvaluator(EvaluationConfig(k_values=[1, 2]))
    report = evaluator.evaluate_single(execution, dataset=dataset)

    assert report.aggregate.num_evaluated == 1
    assert report.aggregate.retrieval["recall@1"] == 1.0
    assert report.aggregate.retrieval["precision@1"] == 1.0
    assert report.aggregate.retrieval["mrr"] == 1.0
    assert report.aggregate.answer["exact_match"] == 1.0
    assert report.per_query[0].retrieved_doc_ids == ["chroma_chunk_001", "chroma_chunk_002"]
