from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from medumm.core.contracts import TaskType


@dataclass(slots=True)
class Artifact:
    kind: str
    path: str
    media_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class InferenceResult:
    request_id: str
    task: TaskType
    model_name: str
    text: str | None = None
    artifacts: list[Artifact] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None

    def __post_init__(self) -> None:
        self.task = TaskType(self.task)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "request_id": self.request_id,
            "task": self.task.value,
            "model_name": self.model_name,
            "text": self.text,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "scores": self.scores,
            "metadata": self.metadata,
            "duration_ms": self.duration_ms,
        }

    @property
    def output_path(self) -> str | None:
        return self.artifacts[0].path if self.artifacts else None


@dataclass(slots=True)
class TrainingResult:
    method: str
    status: str
    output_directory: str
    checkpoint_path: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[Artifact] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "method": self.method,
            "status": self.status,
            "output_directory": self.output_directory,
            "checkpoint_path": self.checkpoint_path,
            "metrics": self.metrics,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class EvaluationResult:
    benchmark: str
    mode: str
    status: str
    dataset_size: int
    output_directory: str
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Artifact] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "benchmark": self.benchmark,
            "mode": self.mode,
            "status": self.status,
            "dataset_size": self.dataset_size,
            "output_directory": self.output_directory,
            "metrics": self.metrics,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "metadata": self.metadata,
        }


def existing_artifacts(result: InferenceResult | TrainingResult | EvaluationResult) -> list[Path]:
    return [Path(item.path) for item in result.artifacts if Path(item.path).is_file()]
