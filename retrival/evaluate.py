"""Offline evaluation CLI for RagDemo using RAG Inspector SDK.

Allows evaluating recorded executions or running interactive questions against
an EvaluationDataset, producing structured metrics and failure diagnoses.

Usage:
    python evaluate.py --dataset path/to/dataset.json --executions path/to/executions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_SDK_SRC = str(Path(_PROJECT_ROOT) / "rag-inspector-sdk" / "src")
if _SDK_SRC not in sys.path:
    sys.path.insert(0, _SDK_SRC)

from rag_inspector import (
    EvaluationConfig,
    EvaluationDataset,
    ExecutionRecord,
    FailureClassifier,
    RAGEvaluator,
)


def load_dataset(dataset_path: Path) -> EvaluationDataset:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    return EvaluationDataset.from_dict(data)


def load_executions(executions_path: Path) -> List[ExecutionRecord]:
    raw = json.loads(executions_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "executions" in raw:
        items = raw["executions"]
    elif isinstance(raw, list):
        items = raw
    else:
        items = [raw]
    return [ExecutionRecord.from_dict(item) for item in items]


def evaluate_records(
    executions: List[ExecutionRecord],
    dataset: EvaluationDataset | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    evaluator = RAGEvaluator(EvaluationConfig())
    report = evaluator.evaluate_executions(executions, dataset=dataset)

    classifier = FailureClassifier()
    diagnoses = []
    for ex in executions:
        diag = classifier.classify(ex, dataset=dataset)
        diagnoses.append({
            "query": ex.question,
            "query_id": ex.query_id,
            "run_id": ex.run_id,
            "diagnosis": diag.to_dict(),
        })

    result = {
        "report": report.to_dict(),
        "diagnoses": diagnoses,
        "summary": report.summary(),
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "evaluation_report.json"
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        md_path = output_dir / "evaluation_report.md"
        md_content = report.summary() + "\n\n### Failure Diagnoses\n\n"
        for d in diagnoses:
            md_content += f"- **{d['query']}**: `{d['diagnosis']['primary']}` - {d['diagnosis']['explanation']}\n"
        md_path.write_text(md_content, encoding="utf-8")
        print(f"Saved reports to {output_dir}")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline executions using RAG Inspector SDK.")
    parser.add_argument("--dataset", type=Path, required=False, help="Path to EvaluationDataset JSON file.")
    parser.add_argument("--executions", type=Path, required=True, help="Path to recorded executions JSON file.")
    parser.add_argument("--output-dir", type=Path, default=Path("./evaluation_output"), help="Output directory for reports.")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset) if args.dataset else None
    executions = load_executions(args.executions)
    res = evaluate_records(executions, dataset=dataset, output_dir=args.output_dir)
    print("\n" + res["summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
