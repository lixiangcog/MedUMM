from __future__ import annotations

from importlib import import_module
from typing import Any

from medumm.core.registry import registry
from medumm.inference.request import InferenceRequest


BUILTIN_BACKBONES = {
    "medical_reference": ("medumm.backbones.medical_reference", "MedicalReferenceAdapter"),
    "medical_linear": ("medumm.backbones.medical_linear", "MedicalLinearAdapter"),
    "medgemma": ("medumm.backbones.medgemma", "MedGemmaAdapter"),
}


def _factory(module_name: str, class_name: str):
    def create():
        return getattr(import_module(module_name), class_name)()

    return create


def register_builtin_backbones() -> None:
    for name, (module_name, class_name) in BUILTIN_BACKBONES.items():
        if not registry.contains("backbone", name):
            registry.add("backbone", name, _factory(module_name, class_name))


class InferencePipeline:
    """Dispatch normalized requests to one lazily imported model adapter."""

    def __init__(
        self,
        backbone_name: str,
        backbone_config: dict[str, Any] | None = None,
    ) -> None:
        register_builtin_backbones()
        self.backbone_name = backbone_name
        self.adapter = registry.get("backbone", backbone_name)()
        self.adapter.load(backbone_config or {})

    def run(self, value: InferenceRequest | dict[str, Any]) -> Any:
        request = InferenceRequest.from_value(value)
        if request.task not in self.adapter.supported_tasks:
            raise NotImplementedError(
                f"{self.backbone_name!r} does not support {request.task!r}."
            )
        method = getattr(self.adapter, request.task)
        if request.task == "understanding":
            return method(
                request.prompt,
                request.images,
                request.videos,
                request.parameters,
            )
        if request.task == "generation":
            return method(request.prompt, request.output_path, request.parameters)
        return method(
            request.prompt,
            request.images,
            request.output_path,
            request.parameters,
        )

    def run_many(
        self,
        values: list[InferenceRequest | dict[str, Any]],
    ) -> list[Any]:
        return [self.run(value) for value in values]

    def close(self) -> None:
        closer = getattr(self.adapter, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> "InferencePipeline":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
