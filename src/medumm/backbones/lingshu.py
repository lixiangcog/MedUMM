from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Any

from medumm.core.contracts import ArchitectureFamily, Modality, ModelCapabilities, TaskType
from medumm.core.interfaces import ModelAdapter
from medumm.core.results import InferenceResult
from medumm.core.runtime import RuntimeContext
from medumm.inference.request import InferenceRequest


DEFAULT_MODEL = "lingshu-medical-mllm/Lingshu-7B"
DEFAULT_SYSTEM_PROMPT = (
    "You are a medical multimodal research assistant. Answer from the supplied "
    "evidence and state when it is insufficient. Do not present output as clinical advice."
)
MUTABLE_REVISIONS = frozenset({"", "main", "master", "latest", "head"})


class LingshuAdapter(ModelAdapter):
    """Native Qwen2.5-VL executor for the Lingshu-7B medical checkpoint."""

    name = "lingshu_7b"
    capabilities = ModelCapabilities(
        tasks=frozenset({TaskType.UNDERSTANDING}),
        input_modalities=frozenset({Modality.TEXT, Modality.IMAGE, Modality.IMAGE_SET}),
        output_modalities=frozenset({Modality.TEXT}),
        architecture=ArchitectureFamily.AUTOREGRESSIVE,
        supports_batching=False,
        max_batch_size=1,
        max_images=4,
        supports_multi_turn=False,
        supports_hidden_states=True,
        notes=(
            "Medical Qwen2.5-VL checkpoint; research use only.",
            "Requires an immutable upstream revision and transformers>=4.52.1.",
        ),
    )

    def load(self, config: dict[str, Any], runtime: RuntimeContext) -> None:
        self.runtime = runtime
        self.model_revision = str(config.get("revision", "")).strip()
        if self.model_revision.casefold() in MUTABLE_REVISIONS:
            raise ValueError(
                "lingshu_7b requires config.revision with an immutable source commit."
            )
        raw_model_path = str(config.get("model_path", DEFAULT_MODEL)).strip()
        candidate = runtime.project_root / raw_model_path
        if not Path(raw_model_path).expanduser().is_absolute() and candidate.exists():
            raw_model_path = str(candidate.resolve())
        self.model_path = raw_model_path
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                "Lingshu requires transformers>=4.52.1 and qwen-vl-utils."
            ) from error

        self.torch = torch
        self.process_vision_info = process_vision_info
        dtype_name = str(config.get("dtype", config.get("torch_dtype", "bfloat16")))
        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            raise ValueError(f"Unknown torch dtype: {dtype_name}")
        load_options: dict[str, Any] = {
            "revision": self.model_revision,
            "dtype": dtype,
            "device_map": config.get("device_map", "auto"),
            "trust_remote_code": bool(config.get("trust_remote_code", False)),
        }
        attention = str(config.get("attn_implementation", "")).strip()
        if attention:
            load_options["attn_implementation"] = attention
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            **load_options,
        ).eval()
        processor_options = {
            key: int(config[key])
            for key in ("min_pixels", "max_pixels")
            if config.get(key) is not None
        }
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            revision=self.model_revision,
            trust_remote_code=bool(config.get("trust_remote_code", False)),
            use_fast=bool(config.get("use_fast_processor", False)),
            **processor_options,
        )
        self.defaults = {
            "max_new_tokens": 64,
            "do_sample": False,
            **dict(config.get("parameters", {})),
        }
        self.system_prompt = str(config.get("system_prompt", DEFAULT_SYSTEM_PROMPT))
        parameter = next(self.model.parameters())
        self.loaded_device = str(parameter.device)
        self.dtype = str(parameter.dtype).removeprefix("torch.")

    def _messages(self, request: InferenceRequest) -> list[dict[str, Any]]:
        if not request.images:
            raise ValueError("Lingshu understanding requires at least one image.")
        if len(request.images) > int(self.capabilities.max_images or 0):
            raise ValueError(
                f"Lingshu accepts at most {self.capabilities.max_images} images per request."
            )
        content = [
            {"type": "image", "image": str(Path(raw_path).resolve())}
            for raw_path in request.images
        ]
        content.append({"type": "text", "text": str(request.prompt or "Describe the images.")})
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def _understand_one(self, request: InferenceRequest) -> InferenceResult:
        messages = self._messages(request)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.loaded_device)
        options = {**self.defaults, **request.parameters}
        if not bool(options.get("do_sample", False)):
            options.pop("temperature", None)
            options.pop("top_p", None)
        uses_cuda = self.loaded_device.startswith("cuda")
        if uses_cuda:
            self.torch.cuda.reset_peak_memory_stats()
            self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **options)
        if uses_cuda:
            self.torch.cuda.synchronize()
        duration_ms = (time.perf_counter() - started) * 1000
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        response = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        peak_memory_mb = None
        if uses_cuda:
            peak_memory_mb = round(self.torch.cuda.max_memory_allocated() / 1024**2, 2)
        return InferenceResult(
            request_id=request.request_id,
            task=TaskType.UNDERSTANDING,
            model_name=self.name,
            text=response,
            metadata={
                "model_id": self.model_path,
                "model_revision": self.model_revision,
                "model_family": "qwen2_5_vl",
                "resource": self.name,
                "source": f"https://huggingface.co/{DEFAULT_MODEL}",
                "device": self.loaded_device,
                "dtype": self.dtype,
                "input_images": len(request.images),
                "generated_tokens": int(trimmed[0].shape[-1]),
                "hostname": platform.node(),
                "scheduler": {
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
                },
                "peak_gpu_memory_mb": peak_memory_mb,
                "clinical_use": False,
                **request.metadata,
            },
            duration_ms=round(duration_ms, 2),
        )

    def understand_batch(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        return [self._understand_one(request) for request in requests]

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "torch") and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
