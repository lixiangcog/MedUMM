from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageOps

from medumm.core.contracts import ArchitectureFamily, Modality, ModelCapabilities, TaskType
from medumm.core.interfaces import ModelAdapter
from medumm.core.io import ensure_directory
from medumm.core.results import Artifact, InferenceResult
from medumm.core.runtime import RuntimeContext
from medumm.inference.request import InferenceRequest


class MedicalReferenceAdapter(ModelAdapter):
    """Deterministic non-clinical adapter for dependency-light workflow tests."""

    name = "medical_reference"
    capabilities = ModelCapabilities(
        tasks=frozenset(TaskType),
        input_modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
        output_modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
        architecture=ArchitectureFamily.REFERENCE,
        supports_batching=True,
        max_batch_size=None,
        max_images=1,
        notes=("Synthetic software reference; not a clinical model.",),
    )

    def load(self, config: dict[str, Any], runtime: RuntimeContext) -> None:
        self.runtime = runtime
        self.seed = int(config.get("seed", 42))
        self.image_size = int(config.get("image_size", 96))
        self.output_directory = Path(config.get("output_directory", "outputs/reference"))
        if not self.output_directory.is_absolute():
            self.output_directory = runtime.project_root / self.output_directory
        if self.image_size < 16:
            raise ValueError("image_size must be at least 16.")

    def _path(self, task: str, prompt: str, output_path: str | None) -> Path:
        if output_path:
            return Path(output_path).expanduser()
        key = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        return self.output_directory / f"{task}_{key}.png"

    def generation(
        self,
        prompt: str | None,
        output_path: str | None,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        text = str(prompt or "synthetic phantom")
        size = int(parameters.get("image_size", self.image_size))
        stable = hashlib.sha256(f"{self.seed}:{text}".encode()).digest()
        generator = random.Random(int.from_bytes(stable[:8], "big"))
        centre_x = size * (0.5 + generator.uniform(-0.08, 0.08))
        centre_y = size * (0.5 + generator.uniform(-0.08, 0.08))
        radius = size * 0.16
        pixels = []
        for y in range(size):
            for x in range(size):
                body = max(0, 1 - math.dist((x, y), (size / 2, size / 2)) / (size / 2))
                spot = math.exp(-((x - centre_x) ** 2 + (y - centre_y) ** 2) / (2 * radius**2))
                pixels.append(max(0, min(255, int(20 + 150 * body + 65 * spot))))
        image = Image.new("L", (size, size))
        image.putdata(pixels)
        path = self._path("generation", text, output_path)
        ensure_directory(path.parent)
        image.save(path)
        return {"task": "generation", "output_path": str(path), "research_only": True}

    def understanding(
        self,
        prompt: str | None,
        images: list[str],
        videos: list[str],
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        del videos
        if "fixed_answer" in parameters:
            response = str(parameters["fixed_answer"])
        elif images:
            with Image.open(Path(images[0])) as source:
                histogram = source.convert("L").histogram()
            mean = sum(index * count for index, count in enumerate(histogram)) / sum(histogram)
            response = str(
                parameters.get("bright_answer", "A")
                if mean >= float(parameters.get("bright_threshold", 96))
                else parameters.get("dark_answer", "B")
            )
        else:
            response = "insufficient evidence"
        return {
            "task": "understanding",
            "understandings": [{"response": response}],
            "prompt": str(prompt or ""),
            "research_only": True,
        }

    def editing(
        self,
        prompt: str | None,
        images: list[str],
        output_path: str | None,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        source_path = Path(images[0])
        with Image.open(source_path) as source:
            image = source.convert("RGB")
        instruction = str(prompt).casefold()
        if "invert" in instruction:
            image = ImageOps.invert(image)
        elif "bright" in instruction:
            image = ImageEnhance.Brightness(image).enhance(float(parameters.get("factor", 1.3)))
        else:
            image = ImageEnhance.Contrast(image).enhance(float(parameters.get("factor", 1.5)))
        path = self._path("editing", instruction, output_path)
        ensure_directory(path.parent)
        image.save(path)
        return {"task": "editing", "output_path": str(path), "research_only": True}

    def understand_batch(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        results = []
        for request in requests:
            output = self.understanding(
                request.prompt, request.images, request.videos, request.parameters
            )
            results.append(
                InferenceResult(
                    request_id=request.request_id,
                    task=TaskType.UNDERSTANDING,
                    model_name=self.name,
                    text=str(output["understandings"][0]["response"]),
                    metadata={"research_only": True, **request.metadata},
                )
            )
        return results

    def generate_batch(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        results = []
        for request in requests:
            output = self.generation(request.prompt, request.output_path, request.parameters)
            results.append(
                InferenceResult(
                    request_id=request.request_id,
                    task=TaskType.GENERATION,
                    model_name=self.name,
                    artifacts=[Artifact("image", output["output_path"], "image/png")],
                    metadata={"research_only": True, **request.metadata},
                )
            )
        return results

    def edit_batch(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        results = []
        for request in requests:
            output = self.editing(
                request.prompt, request.images, request.output_path, request.parameters
            )
            results.append(
                InferenceResult(
                    request_id=request.request_id,
                    task=TaskType.EDITING,
                    model_name=self.name,
                    artifacts=[Artifact("image", output["output_path"], "image/png")],
                    metadata={"research_only": True, **request.metadata},
                )
            )
        return results
