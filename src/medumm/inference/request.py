from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from medumm.core.contracts import TaskType
from medumm.medical.tasks import MedicalTaskType


@dataclass(slots=True)
class InferenceRequest:
    """Model-neutral request shared by all three task pipelines."""

    task: TaskType
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    model: str | None = None
    prompt: str | None = None
    images: list[str] = field(default_factory=list)
    volumes: list[str] = field(default_factory=list)
    videos: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    output_path: str | None = None
    medical_task: MedicalTaskType | None = None

    def __post_init__(self) -> None:
        self.task = TaskType(self.task)
        self.medical_task = MedicalTaskType(self.medical_task) if self.medical_task else None
        self.request_id = str(self.request_id).strip()
        self.model = str(self.model).strip().lower() if self.model else None
        self.images = [str(Path(item).expanduser()) for item in self.images]
        self.volumes = [str(Path(item).expanduser()) for item in self.volumes]
        self.videos = [str(Path(item).expanduser()) for item in self.videos]
        self.validate()

    @classmethod
    def from_value(cls, value: "InferenceRequest | dict[str, Any]") -> "InferenceRequest":
        if isinstance(value, cls):
            value.validate()
            return value
        return cls(
            task=TaskType(value["task"]),
            medical_task=value.get("medical_task"),
            request_id=str(value.get("request_id") or value.get("id") or uuid.uuid4().hex),
            model=value.get("model", value.get("backbone")),
            prompt=value.get("prompt"),
            images=list(value.get("images", [])),
            volumes=list(value.get("volumes", [])),
            videos=list(value.get("videos", [])),
            parameters=dict(value.get("parameters", value.get("params", {}))),
            metadata=dict(value.get("metadata", {})),
            output_path=value.get("output_path"),
        )

    def resolved(self, project_root: str | Path) -> "InferenceRequest":
        """Return a path-resolved copy without mutating the caller's request."""

        root = Path(project_root)

        def resolve(raw_path: str) -> str:
            path = Path(raw_path)
            return str(path if path.is_absolute() else root / path)

        return replace(
            self,
            images=[resolve(path) for path in self.images],
            volumes=[resolve(path) for path in self.volumes],
            videos=[resolve(path) for path in self.videos],
            output_path=resolve(self.output_path) if self.output_path else None,
            parameters=dict(self.parameters),
            metadata=dict(self.metadata),
        )

    def validate(self) -> None:
        if not self.request_id:
            raise ValueError("request_id cannot be empty.")
        if self.task is TaskType.GENERATION and not self.prompt:
            raise ValueError("Generation requires a prompt.")
        if self.task is TaskType.UNDERSTANDING and not (
            self.prompt or self.images or self.volumes or self.videos
        ):
            raise ValueError("Understanding requires text, image, volume, or video input.")
        if self.task is TaskType.EDITING and not (self.prompt and self.images):
            raise ValueError("Editing requires a prompt and at least one image.")
        if self.medical_task is not None and self.task is not TaskType.UNDERSTANDING:
            raise ValueError(
                "Medical semantic tasks currently require the text-output understanding pipeline."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model": self.model,
            "task": self.task.value,
            "medical_task": self.medical_task.value if self.medical_task else None,
            "prompt": self.prompt,
            "images": self.images,
            "volumes": self.volumes,
            "videos": self.videos,
            "parameters": self.parameters,
            "metadata": self.metadata,
            "output_path": self.output_path,
        }
