"""MedUMM public package."""

__version__ = "0.8.0"

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
from medumm.api import catalog, evaluate, infer, post_train, resources
from medumm.evaluation import EvaluationProtocol, MedicalTaskProtocol, create_metric_suite
from medumm.medical import MedicalTaskSample, MedicalTaskType

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
    "MedicalTaskProtocol",
    "MedicalTaskSample",
    "MedicalTaskType",
    "TaskType",
    "TrainingResult",
    "catalog",
    "create_metric_suite",
    "evaluate",
    "infer",
    "post_train",
    "resources",
    "__version__",
]
