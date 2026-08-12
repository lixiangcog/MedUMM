"""MedUMM public package."""

__version__ = "0.4.0"

from medumm.core import (
    EvaluationMode,
    EvaluationResult,
    InferenceResult,
    MetricSuite,
    ModelCapabilities,
    RuntimeContext,
    TaskType,
    TrainingResult,
)
from medumm.inference import InferencePipeline, InferenceRequest
from medumm.api import catalog, evaluate, infer, post_train
from medumm.evaluation import EvaluationProtocol, create_metric_suite

__all__ = [
    "EvaluationMode",
    "EvaluationResult",
    "InferencePipeline",
    "InferenceRequest",
    "InferenceResult",
    "MetricSuite",
    "ModelCapabilities",
    "RuntimeContext",
    "EvaluationProtocol",
    "TaskType",
    "TrainingResult",
    "catalog",
    "create_metric_suite",
    "evaluate",
    "infer",
    "post_train",
    "__version__",
]
