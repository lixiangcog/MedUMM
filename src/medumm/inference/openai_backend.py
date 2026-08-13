from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from medumm.core.contracts import ArchitectureFamily, Modality, ModelCapabilities, TaskType
from medumm.core.interfaces import ModelAdapter
from medumm.core.results import InferenceResult
from medumm.core.runtime import RuntimeContext
from medumm.inference.backends import BackendConfig, InferenceBackend
from medumm.inference.request import InferenceRequest


class OpenAIHTTPAdapter(ModelAdapter):
    """OpenAI-compatible text/VLM client for vLLM and SGLang servers."""

    name = "openai_http"
    capabilities = ModelCapabilities(
        tasks=frozenset({TaskType.UNDERSTANDING}),
        input_modalities=frozenset({Modality.TEXT, Modality.IMAGE, Modality.IMAGE_SET}),
        output_modalities=frozenset({Modality.TEXT}),
        architecture=ArchitectureFamily.AUTOREGRESSIVE,
        supports_batching=True,
        max_batch_size=None,
        max_images=None,
        supported_backends=frozenset({"vllm", "sglang"}),
        supports_continuous_batching=True,
        parallelism=frozenset(
            {"tensor_parallel", "pipeline_parallel", "data_parallel"}
        ),
        notes=("OpenAI-compatible vLLM/SGLang serving adapter.",),
    )

    def load(self, config: dict[str, Any], runtime: RuntimeContext) -> None:
        self.runtime = runtime
        self.backend = BackendConfig.from_dict(config.get("backend"))
        if self.backend.name not in {InferenceBackend.VLLM, InferenceBackend.SGLANG}:
            raise ValueError("OpenAIHTTPAdapter requires a vllm or sglang backend.")
        if self.backend.endpoint is None:
            raise ValueError("OpenAIHTTPAdapter requires backend.endpoint.")
        self.model = str(config.get("model", config.get("model_path", ""))).strip()
        if not self.model:
            raise ValueError("OpenAIHTTPAdapter requires config.model or config.model_path.")
        self.endpoint = self.backend.endpoint.rstrip("/")
        self.system_prompt = str(config.get("system_prompt", "")).strip()
        self.defaults = dict(config.get("parameters", {}))

    @staticmethod
    def _image_content(path: str) -> dict[str, Any]:
        raw = Path(path)
        if not raw.is_file():
            raise FileNotFoundError(raw)
        import base64
        import mimetypes

        media_type = mimetypes.guess_type(raw.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(raw.read_bytes()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{encoded}"},
        }

    def _payload(self, request: InferenceRequest) -> dict[str, Any]:
        # A plain string is the most portable OpenAI representation for
        # text-only requests.  Some VLM chat templates reject a one-element
        # multimodal content list even though image-bearing requests require it.
        if request.images:
            content: str | list[dict[str, Any]] = [
                self._image_content(path) for path in request.images
            ]
            content.append({"type": "text", "text": str(request.prompt or "")})
        else:
            content = str(request.prompt or "")
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": content})
        parameters = {**self.defaults, **request.parameters}
        unsupported = {"classifier_free_guidance", "cfg_schedule"} & set(parameters)
        if unsupported:
            raise ValueError(
                "OpenAI HTTP does not expose Emu3.5 CFG controls; use its patched "
                "vLLM in-process backend."
            )
        return {"model": self.model, "messages": messages, **parameters}

    def _request(self, request: InferenceRequest) -> InferenceResult:
        body = json.dumps(self._payload(request)).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.backend.api_key_environment:
            value = os.environ.get(self.backend.api_key_environment)
            if not value:
                raise RuntimeError(
                    f"Missing API key environment variable {self.backend.api_key_environment}."
                )
            headers["Authorization"] = f"Bearer {value}"
        target = self.endpoint
        if not target.endswith("/chat/completions"):
            target += "/v1/chat/completions"
        http_request = urllib.request.Request(target, data=body, headers=headers, method="POST")
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(
                http_request, timeout=self.backend.request_timeout_seconds
            ) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Inference server returned HTTP {error.code}: {detail}") from error
        choices = payload.get("choices", [])
        if not choices:
            raise RuntimeError("Inference server response contained no choices.")
        text = str(choices[0].get("message", {}).get("content", "")).strip()
        usage = payload.get("usage", {}) or {}
        duration_ms = (time.perf_counter() - started) * 1000
        return InferenceResult(
            request_id=request.request_id,
            task=TaskType.UNDERSTANDING,
            model_name=self.name,
            text=text,
            metadata={
                "backend": self.backend.to_dict(),
                "server_model": payload.get("model", self.model),
                "generated_tokens": usage.get("completion_tokens"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "clinical_use": False,
                **request.metadata,
            },
            duration_ms=round(duration_ms, 3),
        )

    def understand_batch(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        if len(requests) == 1 or not self.backend.continuous_batching:
            return [self._request(request) for request in requests]
        workers = min(len(requests), self.backend.scheduler.max_num_seqs)
        # Concurrent arrivals let the vLLM/SGLang server continuously admit
        # requests between decode steps. executor.map preserves input order.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(self._request, requests))

    def runtime_info(self) -> dict[str, Any]:
        return {
            "backend": self.backend.to_dict(),
            "endpoint": self.endpoint,
            "served_model": self.model,
        }
