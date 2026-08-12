from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from medumm.core.contracts import ArchitectureFamily, Modality, ModelCapabilities, TaskType
from medumm.core.interfaces import ModelAdapter
from medumm.core.results import InferenceResult
from medumm.core.runtime import RuntimeContext
from medumm.inference.request import InferenceRequest
from medumm.medical.linear import featurize, load_model


class MedicalLinearAdapter(ModelAdapter):
    name = "medical_linear"
    capabilities = ModelCapabilities(
        tasks=frozenset({TaskType.UNDERSTANDING}),
        input_modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
        output_modalities=frozenset({Modality.TEXT}),
        architecture=ArchitectureFamily.REFERENCE,
        supports_batching=True,
        max_batch_size=None,
        max_images=8,
        notes=("Engineering baseline; not a clinical model.",),
    )

    def load(self, config: dict[str, Any], runtime: RuntimeContext) -> None:
        self.runtime = runtime
        if not config.get("model_path"):
            raise ValueError("medical_linear requires model_path.")
        model_path = str(config["model_path"])
        candidate = self.runtime.project_root / model_path
        if not Path(model_path).is_absolute() and candidate.exists():
            model_path = str(candidate)
        self.weights, self.bias, manifest = load_model(model_path)
        self.labels = [str(label) for label in manifest["labels"]]
        self.text_dimensions = int(manifest["text_dimensions"])

    def understanding(
        self,
        prompt: str | None,
        images: list[str],
        videos: list[str],
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        del videos, parameters
        features = featurize(str(prompt or ""), images, self.text_dimensions)
        logits = features @ self.weights + self.bias
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        index = int(probabilities.argmax())
        return {
            "task": "understanding",
            "understandings": [{
                "response": self.labels[index],
                "confidence": float(probabilities[index]),
            }],
            "research_only": True,
        }

    def understand_batch(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        results = []
        for request in requests:
            output = self.understanding(
                request.prompt, request.images, request.videos, request.parameters
            )
            answer = output["understandings"][0]
            results.append(
                InferenceResult(
                    request_id=request.request_id,
                    task=TaskType.UNDERSTANDING,
                    model_name=self.name,
                    text=str(answer["response"]),
                    scores={"confidence": float(answer["confidence"])},
                    metadata={"research_only": True, **request.metadata},
                )
            )
        return results
