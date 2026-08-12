from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from medumm.core.contracts import ArchitectureFamily, Modality, ModelCapabilities, TaskType
from medumm.core.interfaces import ModelAdapter
from medumm.core.results import InferenceResult
from medumm.core.runtime import RuntimeContext
from medumm.inference.request import InferenceRequest


SYSTEM_PROMPT = (
    "You are a medical imaging research assistant. Use only the supplied evidence. "
    "State when evidence is insufficient. Do not present output as clinical advice."
)


class MedGemmaAdapter(ModelAdapter):
    """Optional Transformers adapter for image-text medical understanding."""

    name = "medgemma"
    capabilities = ModelCapabilities(
        tasks=frozenset({TaskType.UNDERSTANDING}),
        input_modalities=frozenset({
            Modality.TEXT,
            Modality.IMAGE,
            Modality.IMAGE_SET,
            Modality.VOLUME,
        }),
        output_modalities=frozenset({Modality.TEXT}),
        architecture=ArchitectureFamily.AUTOREGRESSIVE,
        supports_batching=False,
        max_batch_size=1,
        max_images=None,
        supports_multi_turn=False,
        supports_hidden_states=True,
        notes=(
            "Gated weights with separate Health AI Developer Foundations terms.",
            "Outputs require independent verification and are not clinical advice.",
        ),
    )

    def load(self, config: dict[str, Any], runtime: RuntimeContext) -> None:
        self.runtime = runtime
        try:
            import torch
            from transformers import pipeline
        except ModuleNotFoundError as error:
            raise RuntimeError("Install MedUMM with the 'medical' extra for MedGemma.") from error
        model_path = str(config.get("model_path", "google/medgemma-1.5-4b-it"))
        dtype_name = str(config.get("torch_dtype", "bfloat16"))
        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            raise ValueError(f"Unknown torch dtype: {dtype_name}")
        self.system_prompt = str(config.get("system_prompt", SYSTEM_PROMPT))
        self.defaults = dict(config.get("parameters", {"max_new_tokens": 128, "do_sample": False}))
        self.generator = pipeline(
            "image-text-to-text",
            model=model_path,
            torch_dtype=dtype,
            device_map=config.get("device_map", "auto"),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
        )

    @staticmethod
    def _response(value: Any) -> str:
        if isinstance(value, list) and value:
            return MedGemmaAdapter._response(value[0])
        if isinstance(value, dict):
            generated = value.get("generated_text", "")
            if isinstance(generated, list):
                for message in reversed(generated):
                    if isinstance(message, dict) and isinstance(message.get("content"), str):
                        return message["content"].strip()
            return str(generated).strip()
        return str(value).strip()

    def understanding(
        self,
        prompt: str | None,
        images: list[str],
        videos: list[str],
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        del videos
        content = []
        for raw_path in images:
            with Image.open(Path(raw_path)) as image:
                content.append({"type": "image", "image": image.convert("RGB")})
        content.append({"type": "text", "text": str(prompt or "")})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": content},
        ]
        options = {**self.defaults, **parameters}
        return {"understandings": [{"response": self._response(self.generator(text=messages, **options))}]}

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
                    metadata={"clinical_use": False, **request.metadata},
                )
            )
        return results
