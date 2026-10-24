from .errors import (
    InvalidAnswerError,
    InvalidDocumentError,
    InvalidQuestionError,
    InvalidRetrievalError,
    MissingGenerationError,
    MissingPipelineDataError,
    MissingQuestionError,
    MissingRetrievalError,
    RunStateError,
)
from .inspector import RAGInspector
from .run import RAGRun
from .schemas import DocumentRecord, ExecutionRecord, RunStatus

__version__ = "0.1.0"

__all__ = [
    "RAGInspector",
    "RAGRun",
    "DocumentRecord",
    "ExecutionRecord",
    "RunStatus",
    "InvalidDocumentError",
    "InvalidQuestionError",
    "InvalidRetrievalError",
    "InvalidAnswerError",
    "MissingPipelineDataError",
    "MissingQuestionError",
    "MissingRetrievalError",
    "MissingGenerationError",
    "RunStateError",
]

# Evaluation API
try:
    from .evaluation import (
        AbstentionEvaluator,
        AggregateMetrics,
        AnswerCorrectnessEvaluator,
        EvaluationConfig,
        EvaluationDataset,
        EvaluationReport,
        FaithfulnessEvaluator,
        LabeledQuery,
        PerQueryResult,
        RAGEvaluator,
        RelevanceJudgment,
        RetrievalEvaluator,
        average_precision_at_k,
        compute_latency_statistics,
        dcg_at_k,
        mean_average_precision,
        mean_reciprocal_rank,
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
        rouge_l,
    )

    __all__.extend(
        [
            "EvaluationConfig",
            "EvaluationDataset",
            "EvaluationReport",
            "LabeledQuery",
            "PerQueryResult",
            "AggregateMetrics",
            "RAGEvaluator",
            "RetrievalEvaluator",
            "AnswerCorrectnessEvaluator",
            "FaithfulnessEvaluator",
            "AbstentionEvaluator",
            "RelevanceJudgment",
            "recall_at_k",
            "precision_at_k",
            "mean_reciprocal_rank",
            "average_precision_at_k",
            "mean_average_precision",
            "dcg_at_k",
            "ndcg_at_k",
            "rouge_l",
            "compute_latency_statistics",
        ]
    )
except Exception:
    pass

# Extended modules
try:
    from . import annotation, benchmark, comparison, diagnostics, experiments, judge
    from .annotation import (
        AnnotationRecord,
        AnnotationSet,
        agreement_report,
        cohens_kappa,
        inter_rater_agreement,
    )
    from .benchmark import BenchmarkResult, BenchmarkRunner
    from .comparison import VersionComparison, compare_versions
    from .diagnostics import (
        FailureClassifier,
        FailureDiagnosis,
        FailureType,
        classification_report,
        confusion_matrix,
    )
    from .experiments import ExperimentConfig, ExperimentRunner
    from .judge import (
        JudgeProvider,
        JudgeResult,
        JudgeStabilityExperiment,
        JudgeStabilityReport,
        RuleBasedJudgeProvider,
    )

    __all__.extend(
        [
            "annotation",
            "benchmark",
            "comparison",
            "diagnostics",
            "experiments",
            "judge",
            "AnnotationRecord",
            "AnnotationSet",
            "agreement_report",
            "cohens_kappa",
            "inter_rater_agreement",
            "BenchmarkResult",
            "BenchmarkRunner",
            "VersionComparison",
            "compare_versions",
            "FailureClassifier",
            "FailureDiagnosis",
            "FailureType",
            "classification_report",
            "confusion_matrix",
            "ExperimentConfig",
            "ExperimentRunner",
            "JudgeProvider",
            "JudgeResult",
            "RuleBasedJudgeProvider",
            "JudgeStabilityExperiment",
            "JudgeStabilityReport",
        ]
    )
except Exception:
    pass
