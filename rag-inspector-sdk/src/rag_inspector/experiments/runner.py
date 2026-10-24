from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..evaluation.dataset import EvaluationDataset
from ..evaluation.orchestrator import RAGEvaluator


@dataclass
class ExperimentConfig:
    experiment_id: str
    dataset_id: str
    dataset_version: str = "1.0"
    annotation_version: str = "1.0"
    rag_version: str = "v1"
    evaluator_config: dict[str, Any] = field(default_factory=dict)
    judge_config: dict[str, Any] = field(default_factory=dict)
    random_seeds: dict[str, int] = field(default_factory=dict)
    metric_definitions: dict[str, str] = field(default_factory=dict)
    ranking_depth: int = 10
    metadata: dict[str, Any] = field(default_factory=dict)


class ExperimentRunner:
    def __init__(self, output_dir: str | Path = "./experiment_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_experiment(self, config: ExperimentConfig, executions, dataset: EvaluationDataset | None = None):
        start = time.time()
        evaluator = RAGEvaluator()
        report = evaluator.evaluate_executions(executions, dataset=dataset)
        result = {
            "experiment_id": config.experiment_id,
            "config": asdict(config),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "report": report.to_dict(),
            "duration_seconds": time.time() - start,
        }
        out_path = self.output_dir / f"{config.experiment_id}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        # also write markdown
        md_path = self.output_dir / f"{config.experiment_id}.md"
        md_path.write_text(report.summary() + "\n\n" + json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        return {"json": str(out_path), "md": str(md_path), "result": result}
