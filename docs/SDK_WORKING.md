# RAG Inspector SDK: Working Developer Documentation

## Overview
The **RAG Inspector SDK** (`rag-inspector-sdk`) is an independent, framework-agnostic Python SDK for instrumenting, evaluating, diagnosing, benchmarking, and comparing Retrieval-Augmented Generation (RAG) systems.

The core SDK operates in-memory with zero external service dependencies. Optional analytics modules provide diagnostic classification, inter-annotator agreement, paired version regression detection with bootstrap confidence intervals, judge stability analysis, performance benchmarking, and experiment persistence.

---

## Architecture & Data Flow

```
Application Query
       │
       ▼
RAGInspector.start_run(question, query_id=..., metadata=...)
       │
       ▼
RAGRun.record_retrieval(documents, latency_ms=...)
       │
       ▼
RAGRun.record_generation(answer, context=..., latency_ms=...)
       │
       ▼
RAGRun.finish() ──► ExecutionRecord (Immutable)
                          │
         ┌────────────────┴────────────────┐
         ▼                                 ▼
RAGEvaluator.evaluate_executions(...)   FailureClassifier().classify(...)
         │                                 │
         ▼                                 ▼
EvaluationReport (Metrics & Latency)    FailureDiagnosis (Root cause & evidence)
```

---

## 1. Execution Tracing API

### Starting a Tracing Run
```python
from rag_inspector import RAGInspector

inspector = RAGInspector()
run = inspector.start_run(
    question="What are the admission requirements?",
    query_id="q_1001",
    rag_version="v2.1",
    metadata={"embed_model": "all-MiniLM-L6-v2", "top_k": 5}
)
```

### Recording Retrieval
`record_retrieval()` accepts a sequence of strings or dictionaries:
```python
# Pass retrieved chunks preserving stable IDs, text, scores, and ranks
run.record_retrieval(
    [
        {
            "id": "chunk_482",
            "text": "Admission requires a bachelor's degree with a minimum 3.0 GPA.",
            "score": 0.184,
            "rank": 1,
            "metadata": {"file_name": "admissions.pdf", "page_number": 4},
        },
        {
            "id": "chunk_483",
            "text": "Prerequisites include linear algebra and multivariate calculus.",
            "score": 0.291,
            "rank": 2,
            "metadata": {"file_name": "admissions.pdf", "page_number": 5},
        },
    ],
    latency_ms=42.5
)
```

### Recording Generation
```python
run.record_generation(
    answer="Admission requires a bachelor's degree with a 3.0 GPA and math prerequisites.",
    context="Exact prompt context passed to the LLM.",
    latency_ms=280.0
)
```

### Finalizing the Run
```python
execution = run.finish()
print(execution.to_dict())
```

### Handling Failures
If an unhandled exception occurs in the host retriever or generator:
```python
try:
    # generation call...
    pass
except Exception as exc:
    run.record_failure(exc)
```
This preserves all previously recorded telemetry with status `RunStatus.FAILED`.

---

## 2. Evaluation Engine API

### Defining an Evaluation Dataset
```python
from rag_inspector.evaluation import EvaluationDataset, RelevanceJudgment

dataset = EvaluationDataset(name="admissions_eval")
dataset.add_judgment(
    RelevanceJudgment(
        query="What are the admission requirements?",
        query_id="q_1001",
        relevant_doc_ids={"chunk_482", "chunk_483"},
        relevance_scores={"chunk_482": 3.0, "chunk_483": 2.0},
        expected_answer="A bachelor's degree (3.0 GPA), linear algebra, and calculus.",
        metadata={"answerable": True}
    )
)
```

### Running Evaluation
```python
from rag_inspector.evaluation import RAGEvaluator, EvaluationConfig

config = EvaluationConfig(
    k_values=[1, 3, 5, 10],
    precision_denominator="k",  # standard Cranfield cutoff
    compute_retrieval=True,
    compute_answer_correctness=True,
    compute_faithfulness=True,
    compute_abstention=True,
    compute_latency=True,
)

evaluator = RAGEvaluator(config)
report = evaluator.evaluate_single(execution, dataset=dataset)

print(report.summary())
print(report.aggregate.retrieval)   # recall@k, precision@k, mrr, map, ndcg@k
print(report.aggregate.answer)      # exact_match, token_f1, rouge_l_f1
print(report.aggregate.faithfulness)# faithfulness, sentence_support_ratio
print(report.aggregate.latency)     # retrieval/generation mean_ms, p95_ms
```

---

## 3. Failure Diagnosis Engine

```python
from rag_inspector.diagnostics import FailureClassifier

classifier = FailureClassifier()
diagnosis = classifier.classify(execution, dataset=dataset)

print(f"Primary Diagnosis: {diagnosis.primary.value}")
print(f"Explanation: {diagnosis.explanation}")
print(f"Evidence: {diagnosis.evidence}")
```

Supported Diagnosis Types:
- `NO_FAILURE`: Execution succeeded with verified answer and grounded context.
- `WRONG_OR_IRRELEVANT_RETRIEVAL`: Non-relevant documents retrieved.
- `MISSING_RELEVANT_EVIDENCE`: Zero documents retrieved when relevant documents exist.
- `IGNORED_CONTEXT`: Relevant context was supplied to generator, but generation ignored it.
- `UNSUPPORTED_HALLUCINATION`: Answer makes substantive claims unsupported by context.
- `STALE_SOURCE`: Outdated document version used.
- `CONFLICTING_EVIDENCE_MISHANDLED`: Contradictory evidence not resolved.
- `INCOMPLETE_OR_INCORRECT_GENERATION`: Relevant evidence retrieved, but answer incorrect.
- `INCORRECT_ABSTENTION`: System refused to answer despite sufficient evidence.
- `FAILURE_TO_ABSTAIN`: System answered an unanswerable question.
- `UNKNOWN`: Insufficient ground-truth evidence.

---

## 4. Version Comparison API

```python
from rag_inspector.comparison import compare_versions

comparison = compare_versions(
    execs_baseline,
    execs_candidate,
    version_a="v1.0",
    version_b="v2.0",
    dataset=dataset,
    n_boot=1000,
    seed=42
)

print(comparison.summary)
print(comparison.deltas)            # Absolute metric difference
print(comparison.relative_changes)  # Valid relative change (None if baseline=0)
print(comparison.bootstrap_ci)      # 95% paired bootstrap confidence interval
print(comparison.regressions)       # Queries crossing regression thresholds
print(comparison.improvements)      # Queries showing significant improvement
```

---

## 5. Annotator Agreement API

```python
from rag_inspector.annotation import AnnotationRecord, inter_rater_agreement

# Aligns human or LLM judge ratings strictly by query_id
result = inter_rater_agreement(
    annotations_list,
    rater_a="annotator_1",
    rater_b="annotator_2",
    attribute="primary_failure_type"
)

print(f"Raw Agreement: {result['raw_agreement']}")
print(f"Cohen's Kappa: {result['cohens_kappa']}")
print(f"Disagreements: {len(result['disagreements'])}")
```

---

## 6. Judge Stability & Order Bias API

```python
from rag_inspector.judge import JudgeStabilityExperiment, RuleBasedJudgeProvider

exp = JudgeStabilityExperiment(seed=42)

# Compare stability across judges
report = exp.compare_judges(results_judge_a, results_judge_b)

# Test candidate presentation order bias (A vs B position sensitivity)
bias_report = exp.test_pairwise_order_bias(pairwise_fn, test_pairs)
print(f"Order Bias Detected: {bias_report['order_bias_detected']}")
print(f"Inconsistency Rate: {bias_report['inconsistency_rate']}")
```

---

## 7. Performance Benchmarking API

```python
from rag_inspector.benchmark import BenchmarkRunner

runner = BenchmarkRunner()

# Evaluation-only throughput benchmark
res_eval = runner.run_benchmark(executions_list, target_minutes=15.0)
print(f"Throughput: {res_eval.throughput_qps} QPS | Target Met: {res_eval.met_target}")

# End-to-end pipeline benchmark
res_e2e = runner.run_end_to_end_benchmark(app.answer_question, questions_list, target_minutes=15.0)
print(f"E2E Duration: {res_e2e.duration_minutes:.2f} mins")
```
