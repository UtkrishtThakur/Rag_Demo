# RAG Inspector SDK

A lightweight, framework-independent Python SDK for explicitly instrumenting, evaluating, diagnosing, benchmarking, and comparing Retrieval-Augmented Generation (RAG) pipelines.

---

## Key Capabilities

- **Execution Tracing:** Low-overhead, zero-dependency in-memory recording of queries, retrieved passages, supplied prompt contexts, generated answers, and stage latencies.
- **Information Retrieval Metrics:** Mathematically rigorous implementations of **Recall@k**, **Precision@k** (with standard Cranfield cutoff denominator), **MRR**, **MAP**, and **nDCG@k** with hit deduplication.
- **Generation & Faithfulness Evaluation:** Exact Match, Token F1, true LCS-based **ROUGE-L**, categorical correctness verdicts, and lexical/sentence-level contextual faithfulness.
- **Abstention Policy Evaluation:** Correct abstention rate, unnecessary abstention rate, and answer coverage.
- **Evidence-Based Failure Diagnosis:** Multi-class heuristic classification distinguishing verified success (`NO_FAILURE`), retrieval failures (`WRONG_OR_IRRELEVANT_RETRIEVAL`, `MISSING_RELEVANT_EVIDENCE`), generation failures (`UNSUPPORTED_HALLUCINATION`, `IGNORED_CONTEXT`, `INCOMPLETE_OR_INCORRECT_GENERATION`), and abstention issues.
- **Comparative Analysis:** Paired RAG version comparison with absolute deltas, valid relative changes (handling zero baselines), regression threshold alerts, and paired bootstrap 95% confidence intervals.
- **Judge Stability & Order Bias:** Inter-judge agreement, score drift, and pairwise candidate presentation order swapping ($A$ vs $B$ sensitivity).
- **Performance Benchmarking:** Evaluation-only engine throughput and end-to-end RAG latency/QPS benchmarking.

---

## Installation

```bash
pip install ./rag-inspector-sdk
```

For development and testing:

```bash
pip install -e "./rag-inspector-sdk[dev]"
pytest rag-inspector-sdk/tests
```

---

## Quickstart

### 1. Tracing a Pipeline Execution
```python
from rag_inspector import RAGInspector

inspector = RAGInspector()
run = inspector.start_run(
    question="What is RAG?",
    query_id="q_001",
    rag_version="v1.0",
    metadata={"top_k": 5}
)

# Record retrieved passages (strings or dicts with text, id, score, rank)
run.record_retrieval([
    {
        "id": "doc_101",
        "text": "RAG combines retrieval systems with language generation models.",
        "score": 0.12,
        "rank": 1,
        "metadata": {"source": "overview.pdf"}
    }
], latency_ms=25.0)

# Record generated answer and prompt context
run.record_generation(
    answer="RAG combines retrieval with language models.",
    context="RAG combines retrieval systems with language generation models.",
    latency_ms=180.0
)

execution = run.finish()
print(execution.to_dict())
```

### 2. Evaluating Against Ground Truth
```python
from rag_inspector.evaluation import EvaluationDataset, RelevanceJudgment, RAGEvaluator, EvaluationConfig

dataset = EvaluationDataset(name="demo_eval")
dataset.add_judgment(RelevanceJudgment(
    query="What is RAG?",
    query_id="q_001",
    relevant_doc_ids={"doc_101"},
    relevance_scores={"doc_101": 3.0},
    expected_answer="RAG combines retrieval with language models."
))

evaluator = RAGEvaluator(EvaluationConfig(k_values=[1, 3, 5]))
report = evaluator.evaluate_single(execution, dataset=dataset)

print(report.summary())
```

### 3. Diagnosing Failures
```python
from rag_inspector.diagnostics import FailureClassifier

classifier = FailureClassifier()
diagnosis = classifier.classify(execution, dataset=dataset)

print(f"Primary Diagnosis: {diagnosis.primary.value}")
print(f"Explanation: {diagnosis.explanation}")
print(f"Evidence: {diagnosis.evidence}")
```

---

## Error Handling

- `InvalidQuestionError`: Raised when query text is missing, non-string, or empty whitespace.
- `InvalidRetrievalError`: Raised when retrieval payload is not a sequence.
- `InvalidDocumentError`: Raised when a document dictionary lacks non-empty `text`, has invalid `id`/`metadata`, or when a framework-specific object is passed directly without conversion.
- `InvalidAnswerError`: Raised when answer is missing, non-string, or empty whitespace.
- `MissingPipelineDataError`: Raised by `run.finish()` if retrieval or generation was not recorded before finalization.
- `RunStateError`: Raised when mutating or recording to an already completed or failed run.

---

## License

MIT
