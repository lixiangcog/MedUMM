from __future__ import annotations

from typing import Any

import numpy as np

from medumm.medical.linear import featurize, load_model


class MedicalLinearAdapter:
    name = "medical_linear"
    supported_tasks = frozenset({"understanding"})

    def load(self, config: dict[str, Any]) -> None:
        if not config.get("model_path"):
            raise ValueError("medical_linear requires model_path.")
        self.weights, self.bias, manifest = load_model(config["model_path"])
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
