# RAG Inspector SDK + RagDemo: Comprehensive Root-Cause Audit & Resolution

This document records the architectural audit, root causes of identified defects, and resolution actions implemented across the **RAG Inspector SDK** (`rag-inspector-sdk`) and the **RagDemo** host application.

---

## 1. Executive Summary & Defect Inventory

| Defect ID | Subsystem | File Location | Root Cause | Severity | Resolution Status |
|---|---|---|---|---|---|
| **DEF-01** | Version Comparison | `src/rag_inspector/comparison/version_compare.py` | Attempted set intersection on Python dictionaries (`map_a & map_b`), raising `TypeError`. Undefined relative change from zero baseline reported as `0.0` or error. Bootstrap CI not executed. | **Critical** | **Fixed**: Used `set(map_a) & set(map_b)`; represented zero-baseline relative change as `None`; executed paired bootstrap CI; matched by `query_id`. |
| **DEF-02** | Failure Diagnosis | `src/rag_inspector/diagnostics/taxonomy.py` | Classifier forced every execution into a failure category (defaulted to `INCOMPLETE_OR_INCORRECT_GENERATION` even on 100% correct answers). Did not support a success state. | **Critical** | **Fixed**: Added `NO_FAILURE = "no_failure"`. Built evidence-based decision tree verifying answerability, ground-truth match, and context overlap. |
| **DEF-03** | RagDemo Integration | `retrival/retrieval.py`, `retrival/vectordb.py` | `RetrievedChunkView` schema discarded Chroma's `id`. SDK received documents with `id=None`, breaking downstream relevance matching. | **Critical** | **Fixed**: Preserved `id: str` in `RetrievedChunkView`, `_normalize_results`, `record_retrieval`, and `chunks_data`. |
| **DEF-04** | Invalid Test Fixtures | `tests/evaluation/test_evaluator.py`, `tests/comparison/test_compare.py` | Fixtures passed documents with `{"id": "d1"}` lacking required non-empty `text` field, failing validation prematurely. | **High** | **Fixed**: Corrected test fixtures to provide valid `text` while retaining negative tests in `test_run.py`. |
| **DEF-05** | Annotator Agreement | `src/rag_inspector/annotation/agreement.py` | `cohens_kappa()` compared parallel lists strictly by array index, misaligning shuffled or missing ratings. | **High** | **Fixed**: Added `inter_rater_agreement()` aligning strictly by stable `query_id`, handling missing and duplicate annotations. |
| **DEF-06** | Diagnostic Confusion Matrix | `src/rag_inspector/diagnostics/confusion.py` | Dynamically dropped labels not in a provided label set with `pass`. Handled zero denominators inconsistently. | **Medium** | **Fixed**: Deterministic label ordering; dynamically appended observed classes; guarded zero denominators for precision, recall, and macro-F1. |
| **DEF-07** | IR Metrics Consistency | `src/rag_inspector/evaluation/metrics.py` | Recall@k did not deduplicate retrieved document IDs (duplicate IDs could inflate hits). Precision@k denominator was ambiguous. ROUGE-L was a set overlap heuristic. | **High** | **Fixed**: Deduplicated unique relevant hits; capped Recall at 1.0; documented Cranfield denominator $k$; implemented true LCS-based ROUGE-L; integrated nDCG into evaluator. |
| **DEF-08** | Abstention Evaluation | `src/rag_inspector/evaluation/evaluator.py` | Abstention was only a placeholder configuration field without actual rate calculations. | **Medium** | **Fixed**: Implemented `AbstentionEvaluator` computing coverage, correct abstention rate, and unnecessary abstention rate. |
| **DEF-09** | Judge Order Bias | `src/rag_inspector/judge/stability.py` | Conflated dataset item shuffling with pairwise candidate presentation order swapping ($A$ vs $B$). | **Medium** | **Fixed**: Implemented `test_pairwise_order_bias` checking candidate presentation position bias and inconsistency rates. |
| **DEF-10** | Benchmark Scope | `src/rag_inspector/benchmark/runner.py` | Did not distinguish evaluation-only benchmark from end-to-end RAG pipeline benchmark. | **Medium** | **Fixed**: Added `run_end_to_end_benchmark` and distinct `benchmark_mode` tags. |
| **DEF-11** | Docker Dependencies | `requirements.txt`, `Dockerfile` | `frontend/app.py` imported `streamlit`, but `streamlit` was absent from `requirements.txt`. `pytest` was missing from image. | **High** | **Fixed**: Added `streamlit` and `pytest` to `requirements.txt`; configured editable dev install in `Dockerfile`. |
| **DEF-12** | Hygiene & Security | Root directory | Missing `.env.example`, missing comprehensive `.gitignore` and `.dockerignore`. | **Low** | **Fixed**: Added `.env.example`, expanded `.gitignore` and `.dockerignore` to exclude secrets, caches, and local databases. |

---

## 2. Root Cause Analysis by Component

### 2.1 Version Comparison (`comparison/version_compare.py`)
- **Bug:** `map_a & map_b` attempted dictionary bitwise AND, which is invalid syntax in Python.
- **Root Cause:** Conflation of dictionary keys view with dictionary instances.
- **Resolution:** `common_keys = sorted(set(map_a.keys()) & set(map_b.keys()))`.
- **Mathematical Safety:** If baseline metric $\mu_A = 0$, relative change $(\mu_B - \mu_A) / |\mu_A|$ is mathematically undefined. Represented explicitly as `None` rather than fabricating $0.0$ or throwing `ZeroDivisionError`.
- **Paired Bootstrap:** Bootstrapping now resamples paired query differences $(\Delta_i = v_{B,i} - v_{A,i})$ across 1,000 resamples with a fixed seed to yield valid 95% empirical confidence intervals.

### 2.2 Failure Diagnosis (`diagnostics/taxonomy.py`)
- **Bug:** Successful executions where relevant context was retrieved were unconditionally marked as `INCOMPLETE_OR_INCORRECT_GENERATION` if context overlap exceeded 10%.
- **Root Cause:** Incomplete taxonomy decision tree lacking a success termination branch.
- **Resolution:** Added `FailureType.NO_FAILURE`. Introduced explicit rule evaluation:
  1. `answerable is False` with answer $\implies$ `FAILURE_TO_ABSTAIN`.
  2. `answerable is False` with abstention $\implies$ `NO_FAILURE`.
  3. No relevance labels $\implies$ `UNKNOWN` (`unclassified=True`).
  4. 0 relevant docs retrieved $\implies$ `MISSING_RELEVANT_EVIDENCE` (if 0 retrieved) or `WRONG_OR_IRRELEVANT_RETRIEVAL` (if $>0$ retrieved).
  5. Relevant docs retrieved but model abstains $\implies$ `INCORRECT_ABSTENTION`.
  6. Answer matches ground truth or token F1 $\ge 0.7 \implies$ `NO_FAILURE`.
  7. Answer has $< 15\%$ context token overlap on $>8$ words $\implies$ `UNSUPPORTED_HALLUCINATION`.
  8. Supplied context has 0 overlap with answer $\implies$ `IGNORED_CONTEXT`.

### 2.3 RagDemo Document ID Preservation (`retrival/retrieval.py` & `retrival/vectordb.py`)
- **Bug:** `ChromaVectorDB.search()` produced `RetrievedChunk(id=chunk_id, ...)`, but `RetrievedChunkView` in `retrieval.py` omitted `id`.
- **Root Cause:** Lossy normalization intermediate schema.
- **Resolution:** Added `id: str` to `RetrievedChunkView`, preserved chunk ID in `_normalize_results`, passed `id` to SDK `run.record_retrieval(retrieval_docs)`, and maintained ID in output `chunks_data`.

### 2.4 Human Annotations & Kappa (`annotation/`)
- **Bug:** `cohens_kappa()` compared lists by zip order, rendering it vulnerable to shuffled record sequences.
- **Resolution:** Built `inter_rater_agreement()` which indexes ratings by `query_id` before computing agreement and Cohen's Kappa ($\kappa$). Validated `AnnotationRecord` schemas against allowed enum categories.

---
*Audit completed and verified against repository commit `main`.*
