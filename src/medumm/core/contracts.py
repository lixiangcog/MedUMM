from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class TaskType(str, Enum):
    UNDERSTANDING = "understanding"
    GENERATION = "generation"
    EDITING = "editing"


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    IMAGE_SET = "image_set"
    VOLUME = "volume"
    VIDEO = "video"


class ArchitectureFamily(str, Enum):
    AUTOREGRESSIVE = "autoregressive"
    CONTRASTIVE = "contrastive"
    HYBRID = "autoregressive_diffusion_hybrid"
    DIFFUSION = "diffusion"
    REFERENCE = "reference"


class EvaluationMode(str, Enum):
    AUDIT = "audit"
    GENERATE = "generate"
    SCORE = "score"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Machine-readable contract advertised by every model adapter."""

    tasks: frozenset[TaskType]
    input_modalities: frozenset[Modality]
    output_modalities: frozenset[Modality]
    architecture: ArchitectureFamily
    supports_batching: bool = False
    max_batch_size: int | None = None
    max_images: int | None = 1
    supports_multi_turn: bool = False
    supports_hidden_states: bool = False
    supported_backends: frozenset[str] = field(default_factory=lambda: frozenset({"native"}))
    supports_continuous_batching: bool = False
    supports_classifier_free_guidance: bool = False
    parallelism: frozenset[str] = field(default_factory=frozenset)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("A model must support at least one task.")
        if self.max_batch_size is not None and self.max_batch_size < 1:
            raise ValueError("max_batch_size must be positive when provided.")
        if not self.supports_batching and self.max_batch_size not in {None, 1}:
            raise ValueError(
                "A model without batching support cannot advertise max_batch_size > 1."
            )
        if self.max_images is not None and self.max_images < 0:
            raise ValueError("max_images cannot be negative.")
        if not self.supported_backends:
            raise ValueError("A model must advertise at least one inference backend.")
        invalid_parallelism = set(self.parallelism) - {
            "tensor_parallel",
            "pipeline_parallel",
            "data_parallel",
        }
        if invalid_parallelism:
            raise ValueError(
                f"Unknown model parallelism mode(s): {', '.join(sorted(invalid_parallelism))}."
            )

    def supports(self, task: TaskType | str) -> bool:
        return TaskType(task) in self.tasks

    def validate_batch_size(self, batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.max_batch_size is not None and batch_size > self.max_batch_size:
            raise ValueError(
                f"Requested batch size {batch_size} exceeds model limit "
                f"{self.max_batch_size}."
            )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["tasks"] = sorted(item.value for item in self.tasks)
        value["input_modalities"] = sorted(item.value for item in self.input_modalities)
        value["output_modalities"] = sorted(item.value for item in self.output_modalities)
        value["architecture"] = self.architecture.value
        value["supported_backends"] = sorted(self.supported_backends)
        value["parallelism"] = sorted(self.parallelism)
        value["notes"] = list(self.notes)
        return value


@dataclass(frozen=True, slots=True)
class ComponentDescriptor:
    kind: str
    name: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
