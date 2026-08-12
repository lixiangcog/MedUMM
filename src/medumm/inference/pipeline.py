from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, TypeVar

from medumm.core.builtins import register_builtins
from medumm.core.registry import registry
from medumm.core.results import InferenceResult
from medumm.core.runtime import RuntimeContext
from medumm.inference.request import InferenceRequest
from medumm.inference.task_pipelines import TASK_PIPELINES


Item = TypeVar("Item")


def batched(values: Iterable[Item], batch_size: int) -> Iterator[list[Item]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    bucket: list[Item] = []
    for value in values:
        bucket.append(value)
        if len(bucket) == batch_size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket


class InferencePipeline:
    """Application-facing coordinator over the three task pipelines."""

    def __init__(
        self,
        backbone_name: str,
        backbone_config: dict[str, Any] | None = None,
        *,
        runtime: RuntimeContext | None = None,
    ) -> None:
        register_builtins()
        self.backbone_name = backbone_name.strip().lower()
        self.runtime = runtime or RuntimeContext.create(
            command="inference",
            config_path=None,
        )
        self.adapter = registry.models.create(self.backbone_name)
        self.adapter.load(backbone_config or {}, self.runtime)
        self.capabilities = self.adapter.capabilities
        self._pipelines = {
            task: pipeline_type(self.adapter) for task, pipeline_type in TASK_PIPELINES.items()
        }

    def run(self, value: InferenceRequest | dict[str, Any]) -> InferenceResult:
        return self.run_many([value])[0]

    def run_many(
        self,
        values: list[InferenceRequest | dict[str, Any]],
        *,
        batch_size: int = 1,
    ) -> list[InferenceResult]:
        if not values:
            raise ValueError("Inference requires at least one request.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        requests = [
            InferenceRequest.from_value(value).resolved(self.runtime.project_root)
            for value in values
        ]
        mismatches = sorted({
            request.model
            for request in requests
            if request.model is not None and request.model != self.backbone_name
        })
        if mismatches:
            raise ValueError(
                f"Pipeline model is {self.backbone_name!r}, but request(s) specify: "
                f"{', '.join(mismatches)}."
            )
        identifiers = [request.request_id for request in requests]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Inference request identifiers must be unique.")
        indexed_results: dict[int, InferenceResult] = {}
        for task, task_pipeline in self._pipelines.items():
            selected = [(index, request) for index, request in enumerate(requests) if request.task is task]
            effective_batch_size = batch_size
            if not self.capabilities.supports_batching:
                effective_batch_size = 1
            elif self.capabilities.max_batch_size is not None:
                effective_batch_size = min(batch_size, self.capabilities.max_batch_size)
            for chunk in batched(selected, effective_batch_size):
                indices = [index for index, _ in chunk]
                results = task_pipeline.run([request for _, request in chunk])
                indexed_results.update(zip(indices, results, strict=True))
        return [indexed_results[index] for index in range(len(requests))]

    def close(self) -> None:
        self.adapter.close()

    def __enter__(self) -> "InferencePipeline":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
