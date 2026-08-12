from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

from medumm.core.contracts import ArchitectureFamily, Modality, ModelCapabilities, TaskType
from medumm.core.interfaces import ModelAdapter
from medumm.core.results import InferenceResult
from medumm.core.runtime import RuntimeContext
from medumm.inference.request import InferenceRequest


DEFAULT_MODEL = "microsoft/llava-med-v1.5-mistral-7b"
DEFAULT_SYSTEM_PROMPT = ""


class LlavaMedAdapter(ModelAdapter):
    """Official LLaVA-Med v1.5 adapter for biomedical image understanding."""

    name = "llava_med"
    capabilities = ModelCapabilities(
        tasks=frozenset({TaskType.UNDERSTANDING}),
        input_modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
        output_modalities=frozenset({Modality.TEXT}),
        architecture=ArchitectureFamily.AUTOREGRESSIVE,
        supports_batching=False,
        max_batch_size=1,
        max_images=1,
        supports_multi_turn=False,
        supports_hidden_states=True,
        notes=(
            "Biomedical research model; not for clinical care or clinical decisions.",
            "Requires the official microsoft/LLaVA-Med Python package.",
        ),
    )

    @staticmethod
    def _add_source_path(raw_path: Any, project_root: Path) -> str | None:
        if not raw_path:
            return None
        path = Path(str(raw_path)).expanduser()
        path = path if path.is_absolute() else project_root / path
        if not (path / "llava").is_dir():
            raise FileNotFoundError(
                f"LLaVA-Med source_path must contain the llava package: {path}"
            )
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
        return resolved

    def load(self, config: dict[str, Any], runtime: RuntimeContext) -> None:
        self.runtime = runtime
        self.source_path = self._add_source_path(config.get("source_path"), runtime.project_root)
        try:
            import torch
            from llava.constants import (
                DEFAULT_IMAGE_TOKEN,
                DEFAULT_IM_END_TOKEN,
                DEFAULT_IM_START_TOKEN,
                IMAGE_TOKEN_INDEX,
            )
            from llava.conversation import SeparatorStyle, conv_templates
            from llava.mm_utils import (
                KeywordsStoppingCriteria,
                get_model_name_from_path,
                process_images,
                tokenizer_image_token,
            )
            from llava.model.builder import load_pretrained_model
            from llava.utils import disable_torch_init
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                "LLaVA-Med dependencies are unavailable. Install the official "
                "microsoft/LLaVA-Med package or set model.config.source_path to its checkout."
            ) from error

        self.torch = torch
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.SeparatorStyle = SeparatorStyle
        self.conv_templates = conv_templates
        self.KeywordsStoppingCriteria = KeywordsStoppingCriteria
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token

        self.model_path = str(config.get("model_path", DEFAULT_MODEL))
        candidate = runtime.project_root / self.model_path
        if not Path(self.model_path).is_absolute() and candidate.exists():
            self.model_path = str(candidate)
        self.model_revision = str(config.get("revision", "main"))
        self.device = str(config.get("device", "cuda"))
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("LLaVA-Med requires a CUDA device for the v0.3 recipe.")
        self.conversation_mode = str(config.get("conversation_mode", "mistral_instruct"))
        if self.conversation_mode not in conv_templates:
            available = ", ".join(sorted(conv_templates))
            raise ValueError(
                f"Unknown LLaVA-Med conversation_mode {self.conversation_mode!r}; "
                f"available: {available}."
            )
        self.system_prompt = str(config.get("system_prompt", DEFAULT_SYSTEM_PROMPT))
        self.defaults = {
            "max_new_tokens": 64,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": None,
            "num_beams": 1,
            "use_cache": True,
            **dict(config.get("parameters", {})),
        }

        disable_torch_init()
        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_length = (
            load_pretrained_model(
                self.model_path,
                None,
                model_name,
                load_8bit=bool(config.get("load_8bit", False)),
                load_4bit=bool(config.get("load_4bit", False)),
                device_map=config.get("device_map", "auto"),
                device=self.device,
            )
        )
        self.model.eval()
        configured_revision = getattr(self.model.config, "_commit_hash", None)
        if configured_revision:
            self.model_revision = str(configured_revision)
        self.dtype = str(next(self.model.parameters()).dtype).removeprefix("torch.")
        self.loaded_device = str(next(self.model.parameters()).device)

    def _prompt(self, question: str) -> tuple[str, str]:
        image_token = self.DEFAULT_IMAGE_TOKEN
        if getattr(self.model.config, "mm_use_im_start_end", False):
            image_token = (
                self.DEFAULT_IM_START_TOKEN
                + self.DEFAULT_IMAGE_TOKEN
                + self.DEFAULT_IM_END_TOKEN
            )
        conversation = self.conv_templates[self.conversation_mode].copy()
        if self.system_prompt and hasattr(conversation, "system"):
            conversation.system = self.system_prompt
        conversation.append_message(conversation.roles[0], f"{image_token}\n{question}")
        conversation.append_message(conversation.roles[1], None)
        return conversation.get_prompt(), (
            conversation.sep
            if conversation.sep_style != self.SeparatorStyle.TWO
            else conversation.sep2
        )

    def _understand_one(self, request: InferenceRequest) -> InferenceResult:
        if len(request.images) != 1:
            raise ValueError("LLaVA-Med understanding requires exactly one image.")
        prompt, stop_string = self._prompt(str(request.prompt or "Describe this image."))
        input_ids = self.tokenizer_image_token(
            prompt,
            self.tokenizer,
            self.IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)
        with Image.open(Path(request.images[0])) as source:
            image = source.convert("RGB")
        image_tensor = self.process_images(
            [image], self.image_processor, self.model.config
        )[0].unsqueeze(0).to(device=self.device, dtype=self.torch.float16)
        stopping_criteria = self.KeywordsStoppingCriteria(
            [stop_string], self.tokenizer, input_ids
        )
        options = {**self.defaults, **request.parameters}
        if not bool(options.get("do_sample", False)):
            options.pop("temperature", None)
            options.pop("top_p", None)
        if self.device.startswith("cuda"):
            self.torch.cuda.reset_peak_memory_stats()
            self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                stopping_criteria=[stopping_criteria],
                **options,
            )
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        duration_ms = (time.perf_counter() - started) * 1000
        text = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()
        if stop_string and text.endswith(stop_string):
            text = text[: -len(stop_string)].strip()
        peak_memory_mb = None
        if self.device.startswith("cuda"):
            peak_memory_mb = round(self.torch.cuda.max_memory_allocated() / 1024**2, 2)
        return InferenceResult(
            request_id=request.request_id,
            task=TaskType.UNDERSTANDING,
            model_name=self.name,
            text=text,
            metadata={
                "model_id": self.model_path,
                "model_revision": self.model_revision,
                "model_family": "llava_mistral",
                "conversation_mode": self.conversation_mode,
                "device": self.loaded_device,
                "dtype": self.dtype,
                "context_length": self.context_length,
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
