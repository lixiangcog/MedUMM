"""MedUMM public package."""

__version__ = "1.4.0"

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
from medumm.inference import (
    BackendConfig,
    InferenceBackend,
    InferencePipeline,
    InferenceRequest,
    ParallelConfig,
    SchedulerConfig,
    backend_catalog,
)
from medumm.api import catalog, evaluate, infer, post_train, resources
from medumm.evaluation import (
    EvaluationProtocol,
    MedicalCalibrationMetrics,
    MedicalGroundingMetrics,
    MedicalMeasurementMetrics,
    MedicalReportMetrics,
    MedicalTaskProtocol,
    PathologyVQAMetrics,
    create_metric_suite,
)
from medumm.medical import MedicalTaskSample, MedicalTaskType

__all__ = [
    "EvaluationMode",
    "EvaluationResult",
    "InferencePipeline",
    "InferenceBackend",
    "InferenceRequest",
    "InferenceResult",
    "BackendConfig",
    "MetricSuite",
    "ModelCapabilities",
    "RuntimeContext",
    "EvaluationProtocol",
    "MedicalTaskProtocol",
    "PathologyVQAMetrics",
    "MedicalReportMetrics",
    "MedicalGroundingMetrics",
    "MedicalMeasurementMetrics",
    "MedicalCalibrationMetrics",
    "MedicalTaskSample",
    "MedicalTaskType",
    "ParallelConfig",
    "SchedulerConfig",
    "TaskType",
    "TrainingResult",
    "catalog",
    "backend_catalog",
    "create_metric_suite",
    "evaluate",
    "infer",
    "post_train",
    "resources",
    "__version__",
]
