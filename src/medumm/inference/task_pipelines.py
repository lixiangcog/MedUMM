from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from medumm.core.contracts import Modality, TaskType
from medumm.core.exceptions import UnsupportedTaskError
from medumm.core.results import InferenceResult
from medumm.inference.request import InferenceRequest


class TaskPipeline(ABC):
    """Execution-layer pipeline for exactly one multimodal task."""

    task: TaskType

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter

    def validate(self, requests: list[InferenceRequest]) -> None:
        if not requests:
            raise ValueError("A task batch cannot be empty.")
        if not self.adapter.capabilities.supports(self.task):
            raise UnsupportedTaskError(
                f"{self.adapter.name!r} does not support {self.task.value!r}."
            )
        self.adapter.capabilities.validate_batch_size(len(requests))
        max_images = self.adapter.capabilities.max_images
        for request in requests:
            if request.task is not self.task:
                raise ValueError(
                    f"{self.task.value} pipeline received {request.task.value} request."
                )
            if max_images is not None and len(request.images) > max_images:
                raise ValueError(
                    f"{self.adapter.name!r} accepts at most {max_images} image(s), "
                    f"request {request.request_id!r} has {len(request.images)}."
                )
            if request.prompt and Modality.TEXT not in self.adapter.capabilities.input_modalities:
                raise ValueError(f"{self.adapter.name!r} does not accept text input.")
            accepts_images = bool(
                self.adapter.capabilities.input_modalities
                & {Modality.IMAGE, Modality.IMAGE_SET, Modality.VOLUME}
            )
            if request.images and not accepts_images:
                raise ValueError(f"{self.adapter.name!r} does not accept image input.")
            if (
                request.videos
                and Modality.VIDEO not in self.adapter.capabilities.input_modalities
            ):
                raise ValueError(f"{self.adapter.name!r} does not accept video input.")

    def run(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        self.validate(requests)
        results = self.execute(requests)
        if len(results) != len(requests):
            raise RuntimeError(
                f"{self.adapter.name!r} returned {len(results)} result(s) for "
                f"{len(requests)} request(s)."
            )
        for request, result in zip(requests, results, strict=True):
            if result.request_id != request.request_id:
                raise RuntimeError("Adapter changed result ordering or request identifiers.")
            if result.task is not self.task:
                raise RuntimeError("Adapter returned a result for the wrong task.")
        return results

    @abstractmethod
    def execute(self, requests: list[InferenceRequest]) -> list[InferenceResult]: ...


class UnderstandingPipeline(TaskPipeline):
    task = TaskType.UNDERSTANDING

    def execute(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        return self.adapter.understand_batch(requests)


class GenerationPipeline(TaskPipeline):
    task = TaskType.GENERATION

    def execute(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        return self.adapter.generate_batch(requests)


class EditingPipeline(TaskPipeline):
    task = TaskType.EDITING

    def execute(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        return self.adapter.edit_batch(requests)


TASK_PIPELINES = {
    TaskType.UNDERSTANDING: UnderstandingPipeline,
    TaskType.GENERATION: GenerationPipeline,
    TaskType.EDITING: EditingPipeline,
}
