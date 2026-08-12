from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Task = Literal["generation", "editing", "understanding"]


@dataclass(slots=True)
class InferenceRequest:
    task: Task
    prompt: str | None = None
    images: list[str] = field(default_factory=list)
    videos: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    output_path: str | None = None

    @classmethod
    def from_value(cls, value: "InferenceRequest | dict[str, Any]") -> "InferenceRequest":
        if isinstance(value, cls):
            request = value
        else:
            request = cls(
                task=value["task"],
                prompt=value.get("prompt"),
                images=[str(item) for item in value.get("images", [])],
                videos=[str(item) for item in value.get("videos", [])],
                parameters=dict(value.get("parameters", value.get("params", {}))),
                output_path=value.get("output_path"),
            )
        request.validate()
        return request

    def validate(self) -> None:
        if self.task not in {"generation", "editing", "understanding"}:
            raise ValueError(f"Unsupported inference task: {self.task}")
        if self.task == "generation" and not self.prompt:
            raise ValueError("Generation requires a prompt.")
        if self.task == "understanding" and not (self.prompt or self.images):
            raise ValueError("Understanding requires a prompt or an image.")
        if self.task == "editing" and not (self.prompt and self.images):
            raise ValueError("Editing requires a prompt and at least one image.")
