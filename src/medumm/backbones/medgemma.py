from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


SYSTEM_PROMPT = (
    "You are a medical imaging research assistant. Use only the supplied evidence. "
    "State when evidence is insufficient. Do not present output as clinical advice."
)


class MedGemmaAdapter:
    """Optional Transformers adapter for image-text medical understanding."""

    name = "medgemma"
    supported_tasks = frozenset({"understanding"})

    def load(self, config: dict[str, Any]) -> None:
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
