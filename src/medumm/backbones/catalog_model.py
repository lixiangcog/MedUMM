from __future__ import annotations

import json
import os
import platform
import sys
from importlib import import_module
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

from PIL import Image

from medumm.backbones.recipes import MODEL_ADAPTER_RECIPES, ModelExecutor
from medumm.core.contracts import (
    ArchitectureFamily,
    Modality,
    ModelCapabilities,
    TaskType,
)
from medumm.core.interfaces import ModelAdapter
from medumm.core.results import InferenceResult
from medumm.core.runtime import RuntimeContext
from medumm.inference.request import InferenceRequest
from medumm.resources import AccessLevel, MODEL_RESOURCES, ModelRuntimeFamily


SYSTEM_PROMPT = (
    "You are a medical multimodal research assistant. Use only the supplied evidence, "
    "state when evidence is insufficient, and do not present output as clinical advice."
)
_OPEN_CLIP_LOAD_LOCK = Lock()


class CatalogModelAdapter(ModelAdapter):
    """Spec-backed adapter for one cataloged model release.

    Every release selects an explicit executor and prompt protocol in ``recipes.py``.
    Shared executors are used only when the upstream model API is actually compatible;
    repository-specific releases fail closed until their pinned official implementation
    is connected here.
    """

    def __init__(self, resource_name: str) -> None:
        self.spec = MODEL_RESOURCES.get(resource_name)
        self.recipe = MODEL_ADAPTER_RECIPES.get(resource_name)
        self.name = self.spec.name
        architecture = (
            ArchitectureFamily.CONTRASTIVE
            if self.spec.runtime_family
            in {ModelRuntimeFamily.HF_CONTRASTIVE, ModelRuntimeFamily.OPEN_CLIP}
            else ArchitectureFamily.AUTOREGRESSIVE
        )
        self.capabilities = ModelCapabilities(
            tasks=frozenset(self.spec.tasks),
            input_modalities=frozenset(self.spec.input_modalities),
            output_modalities=frozenset({Modality.TEXT}),
            architecture=architecture,
            supports_batching=False,
            max_batch_size=1,
            max_images=self.recipe.max_images,
            supports_hidden_states=True,
            notes=(
                f"catalog_status={self.spec.status.value}",
                f"access={self.spec.access.value}",
                f"runtime_family={self.spec.runtime_family.value}",
                f"executor={self.recipe.executor.value}",
                f"model_type={self.recipe.model_type}",
                "Research use only; outputs require independent verification.",
                *self.spec.notes,
            ),
        )
        self._delegate: ModelAdapter | None = None
        self._model: Any = None
        self._processor: Any = None
        self._pipeline: Any = None

    def load(self, config: dict[str, Any], runtime: RuntimeContext) -> None:
        self.runtime = runtime
        self._enforce_access(config)
        model_path = str(config.get("model_path", self.spec.artifact_id)).strip()
        revision = str(config.get("revision", "")).strip() or None
        if revision and revision.casefold() in {"main", "master", "latest", "head"}:
            raise ValueError(
                f"{self.name} revision must be an immutable source commit, not {revision!r}."
            )
        if not Path(model_path).expanduser().exists() and not revision:
            raise ValueError(
                f"{self.name} requires config.revision with an immutable source commit. "
                "Use a local model_path to load an already pinned snapshot."
            )
        self.model_path = model_path
        self.model_revision = revision
        if self.recipe.executor in {
            ModelExecutor.QWEN2_VL,
            ModelExecutor.QWEN2_5_VL,
            ModelExecutor.QWEN3_VL,
        }:
            self._load_qwen_vl(config, model_path=model_path, revision=revision)
        elif self.recipe.executor is ModelExecutor.CHEXAGENT:
            self._load_chexagent(config, model_path=model_path, revision=revision)
        elif self.recipe.executor is ModelExecutor.INTERNVL_CHAT:
            self._load_internvl_chat(config, model_path=model_path, revision=revision)
        elif self.recipe.executor is ModelExecutor.M3D_LAMED:
            self._load_m3d_lamed(config, model_path=model_path, revision=revision)
        elif self.recipe.executor is ModelExecutor.MEDCLIP:
            self._load_medclip(config)
        elif self.recipe.executor in {
            ModelExecutor.TRANSFORMERS_PIPELINE,
            ModelExecutor.UNIMED_VL,
        }:
            self._load_hf_pipeline(config, model_path=model_path, revision=revision)
        elif self.recipe.executor is ModelExecutor.HF_CONTRASTIVE:
            self._load_hf_contrastive(config, model_path=model_path, revision=revision)
        elif self.recipe.executor is ModelExecutor.OPEN_CLIP_HUB:
            self._load_open_clip(config, model_path=model_path, revision=revision)
        elif self.recipe.executor is ModelExecutor.LLAVA_REPOSITORY:
            self._load_llava_repository(config, runtime)
        else:
            self._load_official_runtime(config, runtime)

    def _enforce_access(self, config: dict[str, Any]) -> None:
        if self.spec.access is not AccessLevel.OPEN and not bool(config.get("accept_terms")):
            raise PermissionError(
                f"{self.name} is {self.spec.access.value}. Review {self.spec.source} and "
                "set config.accept_terms=true only after accepting the upstream terms."
            )

    @staticmethod
    def _torch_dtype(torch: Any, config: dict[str, Any]) -> Any:
        dtype_name = str(config.get("torch_dtype", "bfloat16"))
        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            raise ValueError(f"Unknown torch dtype: {dtype_name}")
        return dtype

    def _load_hf_pipeline(
        self,
        config: dict[str, Any],
        *,
        model_path: str,
        revision: str | None,
    ) -> None:
        try:
            import torch
            from transformers import pipeline
        except ModuleNotFoundError as error:
            raise RuntimeError("Install MedUMM with the 'medical' extra.") from error
        self._pipeline = pipeline(
            "image-text-to-text",
            model=model_path,
            revision=revision,
            torch_dtype=self._torch_dtype(torch, config),
            device_map=config.get("device_map", "auto"),
            trust_remote_code=bool(config.get("trust_remote_code", self.spec.trust_remote_code)),
        )
        self._defaults = dict(
            config.get("parameters", {"max_new_tokens": 128, "do_sample": False})
        )
        self._system_prompt = str(config.get("system_prompt", SYSTEM_PROMPT))

    def _load_qwen_vl(
        self,
        config: dict[str, Any],
        *,
        model_path: str,
        revision: str | None,
    ) -> None:
        try:
            import torch
            import transformers
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                f"{self.name} requires its pinned Transformers environment and qwen-vl-utils."
            ) from error
        model_class_name = str(self.recipe.model_class or "").strip()
        model_class = getattr(transformers, model_class_name, None)
        if model_class is None:
            raise RuntimeError(
                f"The current Transformers installation has no {model_class_name}. "
                f"Use the {self.name} model environment."
            )
        dtype = self._torch_dtype(torch, config)
        load_options: dict[str, Any] = {
            "revision": revision,
            "torch_dtype": dtype,
            "device_map": config.get("device_map", "auto"),
            "trust_remote_code": bool(
                config.get("trust_remote_code", self.spec.trust_remote_code)
            ),
        }
        attention = str(config.get("attn_implementation", "")).strip()
        if attention:
            load_options["attn_implementation"] = attention
        self._model = model_class.from_pretrained(model_path, **load_options).eval()
        processor_options = {
            key: int(config[key])
            for key in ("min_pixels", "max_pixels")
            if config.get(key) is not None
        }
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=bool(
                config.get("trust_remote_code", self.spec.trust_remote_code)
            ),
            **processor_options,
        )
        self._process_vision_info = process_vision_info
        self._torch = torch
        self._defaults = {
            "max_new_tokens": 128,
            "do_sample": False,
            "use_cache": True,
            **dict(config.get("parameters", {})),
        }
        self._system_prompt = str(config.get("system_prompt", SYSTEM_PROMPT))
        parameter = next(self._model.parameters())
        self._device = str(parameter.device)
        self._dtype = str(parameter.dtype).removeprefix("torch.")

    def _load_chexagent(
        self,
        config: dict[str, Any],
        *,
        model_path: str,
        revision: str | None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as error:
            raise RuntimeError(f"{self.name} requires its pinned CheXagent environment.") from error
        trust_remote_code = bool(config.get("trust_remote_code", True))
        self._processor = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            revision=revision,
            device_map=config.get("device_map", "auto"),
            torch_dtype=self._torch_dtype(torch, config),
            trust_remote_code=trust_remote_code,
        ).eval()
        self._torch = torch
        self._defaults = {
            "max_new_tokens": 128,
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
            **dict(config.get("parameters", {})),
        }
        self._system_prompt = str(config.get("system_prompt", SYSTEM_PROMPT))
        parameter = next(self._model.parameters())
        self._device = str(parameter.device)
        self._dtype = str(parameter.dtype).removeprefix("torch.")

    def _load_internvl_chat(
        self,
        config: dict[str, Any],
        *,
        model_path: str,
        revision: str | None,
    ) -> None:
        try:
            import torch
            import torchvision.transforms as transforms
            from torchvision.transforms.functional import InterpolationMode
            from transformers import AutoModel, AutoTokenizer
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(f"{self.name} requires its pinned InternVL environment.") from error
        trust_remote_code = bool(config.get("trust_remote_code", True))
        load_options: dict[str, Any] = {
            "revision": revision,
            "device_map": config.get("device_map", "auto"),
            "torch_dtype": self._torch_dtype(torch, config),
            "low_cpu_mem_usage": True,
            "trust_remote_code": trust_remote_code,
        }
        if config.get("use_flash_attn") is not None:
            load_options["use_flash_attn"] = bool(config["use_flash_attn"])
        self._model = AutoModel.from_pretrained(model_path, **load_options).eval()
        self._processor = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )
        self._torch = torch
        self._internvl_transform = transforms.Compose(
            [
                transforms.Lambda(lambda image: image.convert("RGB")),
                transforms.Resize(
                    (int(config.get("input_size", 448)),) * 2,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        self._internvl_input_size = int(config.get("input_size", 448))
        self._internvl_max_tiles = int(config.get("max_tiles", 12))
        self._defaults = {
            "max_new_tokens": 256,
            "do_sample": False,
            **dict(config.get("parameters", {})),
        }
        self._system_prompt = str(config.get("system_prompt", SYSTEM_PROMPT))
        parameter = next(self._model.parameters())
        self._device = str(parameter.device)
        self._dtype = str(parameter.dtype).removeprefix("torch.")

    def _load_m3d_lamed(
        self,
        config: dict[str, Any],
        *,
        model_path: str,
        revision: str | None,
    ) -> None:
        try:
            import numpy as np
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as error:
            raise RuntimeError(f"{self.name} requires its pinned M3D environment.") from error
        trust_remote_code = bool(config.get("trust_remote_code", True))
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            revision=revision,
            torch_dtype=self._torch_dtype(torch, config),
            device_map=config.get("device_map", "auto"),
            trust_remote_code=trust_remote_code,
        ).eval()
        self._processor = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            model_max_length=int(config.get("model_max_length", 512)),
            padding_side="right",
            use_fast=False,
            trust_remote_code=trust_remote_code,
        )
        self._np = np
        self._torch = torch
        self._m3d_projection_tokens = int(config.get("projection_tokens", 256))
        self._defaults = {
            "max_new_tokens": 256,
            "do_sample": False,
            **dict(config.get("parameters", {})),
        }
        parameter = next(self._model.parameters())
        self._device = str(parameter.device)
        self._dtype = str(parameter.dtype).removeprefix("torch.")

    def _load_medclip(self, config: dict[str, Any]) -> None:
        try:
            import torch
            from medclip import MedCLIPModel, MedCLIPProcessor
            from medclip import MedCLIPVisionModel, MedCLIPVisionModelViT
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError("Install the pinned official MedCLIP environment.") from error
        variant = str(config.get("vision_variant", "vit")).casefold()
        vision_class = MedCLIPVisionModelViT if variant == "vit" else MedCLIPVisionModel
        self._model = MedCLIPModel(vision_cls=vision_class)
        checkpoint_path = Path(str(config.get("checkpoint_path", ""))).expanduser()
        if not checkpoint_path.is_dir():
            raise ValueError("MedCLIP requires checkpoint_path to a pinned extracted checkpoint.")
        self._model.from_pretrained(input_dir=str(checkpoint_path))
        device = str(config.get("device", self.runtime.device))
        self._model = self._model.to(device).eval()
        self._processor = MedCLIPProcessor()
        self._torch = torch
        self._device = device
        self._dtype = str(next(self._model.parameters()).dtype).removeprefix("torch.")

    def _load_hf_contrastive(
        self,
        config: dict[str, Any],
        *,
        model_path: str,
        revision: str | None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ModuleNotFoundError as error:
            raise RuntimeError("Install MedUMM with the 'medical' extra.") from error
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=bool(config.get("trust_remote_code", self.spec.trust_remote_code)),
        )
        load_options: dict[str, Any] = {
            "revision": revision,
            "torch_dtype": self._torch_dtype(torch, config),
            "trust_remote_code": bool(
                config.get("trust_remote_code", self.spec.trust_remote_code)
            ),
        }
        device_map = config.get("device_map")
        if device_map is not None:
            load_options["device_map"] = device_map
        self._model = AutoModel.from_pretrained(model_path, **load_options).eval()
        if device_map is None:
            self._model = self._model.to(str(config.get("device", self.runtime.device)))

    def _load_open_clip(
        self,
        config: dict[str, Any],
        *,
        model_path: str,
        revision: str | None,
    ) -> None:
        try:
            import open_clip
            import torch
        except ModuleNotFoundError as error:
            raise RuntimeError("Install the optional 'open-clip-torch' package.") from error
        model_name = str(config.get("open_clip_model_name", "")).strip()
        checkpoint_value = str(config.get("checkpoint_path", "")).strip()
        checkpoint_path = Path(checkpoint_value).expanduser() if checkpoint_value else None
        device = str(config.get("device", self.runtime.device))
        local_snapshot = Path(model_path).expanduser()
        text_model_value = str(config.get("text_model_path", "")).strip()
        text_model_path = Path(text_model_value).expanduser() if text_model_value else None
        if local_snapshot.is_dir() and (local_snapshot / "open_clip_config.json").is_file():
            if text_model_path is None or not text_model_path.is_dir():
                raise ValueError(
                    "A local OpenCLIP snapshot requires text_model_path for the pinned "
                    "Hugging Face text encoder and tokenizer."
                )
            weights = next(
                (
                    local_snapshot / filename
                    for filename in (
                        "open_clip_model.safetensors",
                        "open_clip_pytorch_model.bin",
                        "pytorch_model.bin",
                    )
                    if (local_snapshot / filename).is_file()
                ),
                None,
            )
            if weights is None:
                raise FileNotFoundError("The local OpenCLIP snapshot has no model checkpoint.")
            factory = import_module("open_clip.factory")
            original_download = factory.download_pretrained_from_hf
            original_get_config = factory._get_hf_config

            def local_download(
                model_id: str,
                filename: str | None = None,
                revision: str | None = None,
                cache_dir: str | None = None,
                **kwargs: Any,
            ) -> str:
                del model_id, revision, cache_dir, kwargs
                if filename == "open_clip_config.json":
                    return str(local_snapshot / filename)
                if filename:
                    candidate = local_snapshot / filename
                    if candidate.is_file():
                        return str(candidate)
                return str(weights)

            def local_get_config(model_id: str, cache_dir: str | None = None) -> dict[str, Any]:
                del model_id, cache_dir
                value = json.loads(
                    (local_snapshot / "open_clip_config.json").read_text(encoding="utf-8")
                )
                text_config = value.setdefault("model_cfg", {}).setdefault("text_cfg", {})
                text_config["hf_model_name"] = str(text_model_path.resolve())
                text_config["hf_tokenizer_name"] = str(text_model_path.resolve())
                return value

            with _OPEN_CLIP_LOAD_LOCK:
                factory.download_pretrained_from_hf = local_download
                factory._get_hf_config = local_get_config
                try:
                    model_name = f"hf-hub:{local_snapshot.resolve()}"
                    self._model, _, self._preprocess = (
                        open_clip.create_model_and_transforms(
                            model_name,
                            device=device,
                            require_pretrained=True,
                        )
                    )
                    self._tokenizer = open_clip.get_tokenizer(model_name)
                finally:
                    factory.download_pretrained_from_hf = original_download
                    factory._get_hf_config = original_get_config
        elif model_name and checkpoint_path is not None and checkpoint_path.is_file():
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=str(checkpoint_path),
                device=device,
            )
            self._tokenizer = open_clip.get_tokenizer(
                model_name,
                cache_dir=config.get("cache_dir"),
            )
        else:
            if local_snapshot.exists():
                raise ValueError(
                    "A local OpenCLIP snapshot requires open_clip_model_name and "
                    "checkpoint_path. Remote hub loading can infer both from the pinned release."
                )
            if not revision:
                raise ValueError("OpenCLIP hub loading requires an immutable revision.")
            model_name = f"hf-hub:{model_path}"
            factory = import_module("open_clip.factory")
            original_download = factory.download_pretrained_from_hf

            def pinned_download(
                model_id: str,
                filename: str | None = None,
                revision: str | None = None,
                cache_dir: str | None = None,
                **kwargs: Any,
            ) -> str:
                del revision
                return original_download(
                    model_id,
                    filename=filename,
                    revision=self.model_revision,
                    cache_dir=cache_dir,
                    **kwargs,
                )

            with _OPEN_CLIP_LOAD_LOCK:
                factory.download_pretrained_from_hf = pinned_download
                try:
                    self._model, _, self._preprocess = (
                        open_clip.create_model_and_transforms(
                            model_name,
                            device=device,
                            cache_dir=config.get("cache_dir"),
                            require_pretrained=True,
                        )
                    )
                    self._tokenizer = open_clip.get_tokenizer(
                        model_name,
                        cache_dir=config.get("cache_dir"),
                    )
                finally:
                    factory.download_pretrained_from_hf = original_download
        self._torch = torch
        self._model.eval()

    def _load_llava_repository(
        self,
        config: dict[str, Any],
        runtime: RuntimeContext,
    ) -> None:
        from medumm.backbones.llava_med import LlavaMedAdapter

        delegate = LlavaMedAdapter()
        delegate_config = dict(config)
        delegate_config.pop("accept_terms", None)
        delegate.load(delegate_config, runtime)
        self._delegate = delegate

    def _load_official_runtime(
        self,
        config: dict[str, Any],
        runtime: RuntimeContext,
    ) -> None:
        del runtime
        source_path = str(config.get("source_path", "")).strip()
        if not source_path:
            raise RuntimeError(
                f"{self.name} uses the built-in {self.recipe.executor.value} adapter over its "
                "pinned official source. Set config.source_path to that checkout; arbitrary "
                "Python bridge classes are no longer part of the public model interface."
            )
        path = Path(source_path).expanduser()
        if not path.is_absolute():
            path = self.runtime.project_root / path
        if not path.is_dir():
            raise FileNotFoundError(f"Official source checkout does not exist: {path}")
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
        entrypoint = str(self.recipe.official_entrypoint or "")
        raise RuntimeError(
            f"{self.name} selected {self.recipe.executor.value} with official entry point "
            f"{entrypoint!r}, but this repository-specific executor is not available in the "
            "current MedUMM build. This is an adapter implementation failure, not a request "
            "for a user-supplied bridge."
        )

    @staticmethod
    def _response(value: Any) -> str:
        if isinstance(value, list) and value:
            return CatalogModelAdapter._response(value[0])
        if isinstance(value, dict):
            generated = value.get("generated_text", "")
            if isinstance(generated, list):
                for message in reversed(generated):
                    if isinstance(message, dict) and isinstance(message.get("content"), str):
                        return message["content"].strip()
            return str(generated).strip()
        return str(value).strip()

    def _hf_generate(self, request: InferenceRequest) -> InferenceResult:
        content: list[dict[str, Any]] = []
        opened: list[Image.Image] = []
        try:
            for raw_path in [*request.images, *request.volumes]:
                image = Image.open(Path(raw_path)).convert("RGB")
                opened.append(image)
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": str(request.prompt or "")})
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self._system_prompt}],
                },
                {"role": "user", "content": content},
            ]
            started = perf_counter()
            output = self._pipeline(text=messages, **{**self._defaults, **request.parameters})
            duration_ms = (perf_counter() - started) * 1000
        finally:
            for image in opened:
                image.close()
        return InferenceResult(
            request_id=request.request_id,
            task=TaskType.UNDERSTANDING,
            model_name=self.name,
            text=self._response(output),
            metadata={
                "resource": self.name,
                "source": self.spec.source,
                "clinical_use": False,
                **request.metadata,
            },
            duration_ms=duration_ms,
        )

    def _qwen_generate(self, request: InferenceRequest) -> InferenceResult:
        if request.volumes:
            raise ValueError(
                f"{self.name} accepts volume studies only after explicit slice conversion."
            )
        if not request.images:
            raise ValueError(f"{self.name} requires at least one image.")
        if self.recipe.max_images is not None and len(request.images) > self.recipe.max_images:
            raise ValueError(
                f"{self.name} accepts at most {self.recipe.max_images} images per request."
            )
        content = [
            {"type": "image", "image": str(Path(raw_path).resolve())}
            for raw_path in request.images
        ]
        content.append({"type": "text", "text": str(request.prompt or "Describe the images.")})
        messages: list[dict[str, Any]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": content})
        rendered = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self._processor(
            text=[rendered],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._device)
        options = {**self._defaults, **request.parameters}
        if not bool(options.get("do_sample", False)):
            options.pop("temperature", None)
            options.pop("top_p", None)
        uses_cuda = self._device.startswith("cuda")
        if uses_cuda:
            self._torch.cuda.reset_peak_memory_stats()
            self._torch.cuda.synchronize()
        started = perf_counter()
        with self._torch.inference_mode():
            generated = self._model.generate(**inputs, **options)
        if uses_cuda:
            self._torch.cuda.synchronize()
        duration_ms = (perf_counter() - started) * 1000
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated, strict=True)
        ]
        text = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return InferenceResult(
            request_id=request.request_id,
            task=TaskType.UNDERSTANDING,
            model_name=self.name,
            text=text,
            metadata={
                "resource": self.name,
                "model_id": self.model_path,
                "model_revision": self.model_revision,
                "model_family": self.recipe.model_type,
                "executor": self.recipe.executor.value,
                "device": self._device,
                "dtype": self._dtype,
                "input_images": len(request.images),
                "generated_tokens": int(trimmed[0].shape[-1]),
                "hostname": platform.node(),
                "scheduler": {
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
                },
                "peak_gpu_memory_mb": (
                    round(self._torch.cuda.max_memory_allocated() / 1024**2, 2)
                    if uses_cuda
                    else None
                ),
                "clinical_use": False,
                **request.metadata,
            },
            duration_ms=round(duration_ms, 2),
        )

    def _chexagent_generate(self, request: InferenceRequest) -> InferenceResult:
        if not request.images:
            raise ValueError("CheXagent requires at least one chest X-ray image.")
        query = self._processor.from_list_format(
            [
                *({"image": str(Path(path).resolve())} for path in request.images),
                {"text": str(request.prompt or "Describe the chest X-ray findings.")},
            ]
        )
        conversation = [
            {"from": "system", "value": self._system_prompt},
            {"from": "human", "value": query},
        ]
        input_ids = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._device)
        options = {**self._defaults, **request.parameters}
        if not bool(options.get("do_sample", False)):
            options.pop("temperature", None)
            options.pop("top_p", None)
        uses_cuda = self._device.startswith("cuda")
        if uses_cuda:
            self._torch.cuda.reset_peak_memory_stats()
            self._torch.cuda.synchronize()
        started = perf_counter()
        with self._torch.inference_mode():
            generated = self._model.generate(input_ids, **options)[0]
        if uses_cuda:
            self._torch.cuda.synchronize()
        duration_ms = (perf_counter() - started) * 1000
        text = self._processor.decode(
            generated[input_ids.shape[-1] :],
            skip_special_tokens=True,
        ).strip()
        return InferenceResult(
            request_id=request.request_id,
            task=TaskType.UNDERSTANDING,
            model_name=self.name,
            text=text,
            metadata={
                "resource": self.name,
                "model_id": self.model_path,
                "model_revision": self.model_revision,
                "model_family": "chexagent",
                "executor": self.recipe.executor.value,
                "device": self._device,
                "dtype": self._dtype,
                "generated_tokens": int(generated.shape[-1] - input_ids.shape[-1]),
                "clinical_use": False,
                **request.metadata,
            },
            duration_ms=round(duration_ms, 2),
        )

    @staticmethod
    def _internvl_ratio(
        width: int,
        height: int,
        *,
        image_size: int,
        max_tiles: int,
    ) -> tuple[int, int]:
        ratios = sorted(
            {
                (columns, rows)
                for count in range(1, max_tiles + 1)
                for columns in range(1, count + 1)
                for rows in range(1, count + 1)
                if 1 <= columns * rows <= max_tiles
            },
            key=lambda value: value[0] * value[1],
        )
        aspect = width / height
        return min(
            ratios,
            key=lambda value: (
                abs(aspect - value[0] / value[1]),
                -int(width * height > 0.5 * image_size**2 * value[0] * value[1]),
            ),
        )

    def _internvl_pixels(self, raw_path: str) -> Any:
        with Image.open(raw_path) as source:
            image = source.convert("RGB")
        columns, rows = self._internvl_ratio(
            *image.size,
            image_size=self._internvl_input_size,
            max_tiles=self._internvl_max_tiles,
        )
        resized = image.resize(
            (columns * self._internvl_input_size, rows * self._internvl_input_size)
        )
        tiles = []
        for index in range(columns * rows):
            left = index % columns * self._internvl_input_size
            top = index // columns * self._internvl_input_size
            tiles.append(
                resized.crop(
                    (
                        left,
                        top,
                        left + self._internvl_input_size,
                        top + self._internvl_input_size,
                    )
                )
            )
        if len(tiles) > 1:
            tiles.append(image.resize((self._internvl_input_size,) * 2))
        return self._torch.stack([self._internvl_transform(tile) for tile in tiles])

    def _internvl_generate(self, request: InferenceRequest) -> InferenceResult:
        if not request.images:
            raise ValueError(f"{self.name} requires at least one image.")
        tensors = [self._internvl_pixels(path) for path in request.images]
        patch_counts = [int(tensor.shape[0]) for tensor in tensors]
        pixels = self._torch.cat(tensors).to(device=self._device, dtype=next(self._model.parameters()).dtype)
        image_prefix = "".join("<image>\n" for _ in request.images)
        question = f"{self._system_prompt}\n{image_prefix}{request.prompt or 'Describe the images.'}".strip()
        options = {**self._defaults, **request.parameters}
        started = perf_counter()
        chat_options: dict[str, Any] = {}
        if len(patch_counts) > 1:
            chat_options["num_patches_list"] = patch_counts
        with self._torch.inference_mode():
            response = self._model.chat(
                self._processor,
                pixels,
                question,
                options,
                **chat_options,
            )
        duration_ms = (perf_counter() - started) * 1000
        if isinstance(response, tuple):
            response = response[0]
        return InferenceResult(
            request_id=request.request_id,
            task=TaskType.UNDERSTANDING,
            model_name=self.name,
            text=str(response).strip(),
            metadata={
                "resource": self.name,
                "model_id": self.model_path,
                "model_revision": self.model_revision,
                "model_family": "internvl_chat",
                "executor": self.recipe.executor.value,
                "device": self._device,
                "dtype": self._dtype,
                "input_images": len(request.images),
                "image_tiles": sum(patch_counts),
                "clinical_use": False,
                **request.metadata,
            },
            duration_ms=round(duration_ms, 2),
        )

    def _m3d_generate(self, request: InferenceRequest) -> InferenceResult:
        if len(request.volumes) != 1:
            raise ValueError("M3D-LaMed requires exactly one normalized .npy volume.")
        volume_path = Path(request.volumes[0])
        if volume_path.suffix.casefold() != ".npy":
            raise ValueError("M3D-LaMed volume input must be a .npy file.")
        volume = self._np.load(volume_path)
        if volume.shape == (32, 256, 256):
            volume = volume[None, ...]
        if volume.shape != (1, 32, 256, 256):
            raise ValueError(
                f"M3D-LaMed expects (1, 32, 256, 256), found {tuple(volume.shape)}."
            )
        prompt = "<im_patch>" * self._m3d_projection_tokens + str(
            request.prompt or "Describe the medically relevant findings."
        )
        input_ids = self._processor(prompt, return_tensors="pt")["input_ids"].to(self._device)
        image_tensor = (
            self._torch.from_numpy(volume).unsqueeze(0).to(
                device=self._device,
                dtype=next(self._model.parameters()).dtype,
            )
        )
        options = {**self._defaults, **request.parameters}
        started = perf_counter()
        with self._torch.inference_mode():
            generated = self._model.generate(image_tensor, input_ids, **options)
        duration_ms = (perf_counter() - started) * 1000
        if isinstance(generated, tuple):
            generated = generated[0]
        text = self._processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return InferenceResult(
            request_id=request.request_id,
            task=TaskType.UNDERSTANDING,
            model_name=self.name,
            text=text,
            metadata={
                "resource": self.name,
                "model_id": self.model_path,
                "model_revision": self.model_revision,
                "model_family": "lamed_phi3",
                "executor": self.recipe.executor.value,
                "device": self._device,
                "dtype": self._dtype,
                "volume_shape": list(volume.shape),
                "clinical_use": False,
                **request.metadata,
            },
            duration_ms=round(duration_ms, 2),
        )

    @staticmethod
    def _candidates(request: InferenceRequest) -> list[str]:
        values = request.parameters.get("candidates")
        if not isinstance(values, list) or len(values) < 2:
            raise ValueError(
                "Contrastive model requests require parameters.candidates with at least two labels."
            )
        return [str(item) for item in values]

    def _hf_rank(self, request: InferenceRequest) -> InferenceResult:
        import torch

        if len(request.images) != 1:
            raise ValueError("Contrastive candidate ranking requires exactly one image.")
        candidates = self._candidates(request)
        with Image.open(request.images[0]) as image:
            inputs = self._processor(
                text=candidates,
                images=image.convert("RGB"),
                return_tensors="pt",
                padding=True,
            )
        device = next(self._model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        uses_cuda = str(device).startswith("cuda")
        if uses_cuda:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = perf_counter()
        with torch.inference_mode():
            outputs = self._model(**inputs)
            logits = getattr(outputs, "logits_per_image", None)
            if logits is None:
                raise RuntimeError("The contrastive model did not return logits_per_image.")
            probabilities = logits[0].float().softmax(dim=-1).cpu().tolist()
        if uses_cuda:
            torch.cuda.synchronize(device)
        duration_ms = (perf_counter() - started) * 1000
        scores = {candidate: float(score) for candidate, score in zip(candidates, probabilities)}
        prediction = max(scores, key=scores.get)
        peak_memory_mb = (
            round(torch.cuda.max_memory_allocated(device) / 1024**2, 2)
            if uses_cuda
            else None
        )
        return InferenceResult(
            request_id=request.request_id,
            task=TaskType.UNDERSTANDING,
            model_name=self.name,
            text=prediction,
            scores=scores,
            metadata={
                "resource": self.name,
                "model_id": self.model_path,
                "model_revision": self.model_revision,
                "model_family": "hf_contrastive",
                "executor": self.recipe.executor.value,
                "device": str(device),
                "dtype": str(next(self._model.parameters()).dtype).removeprefix("torch."),
                "peak_gpu_memory_mb": peak_memory_mb,
                "hostname": platform.node(),
                "scheduler": {
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
                },
                "clinical_use": False,
                **request.metadata,
            },
            duration_ms=round(duration_ms, 2),
        )

    def _open_clip_rank(self, request: InferenceRequest) -> InferenceResult:
        if len(request.images) != 1:
            raise ValueError("Contrastive candidate ranking requires exactly one image.")
        candidates = self._candidates(request)
        device = str(next(self._model.parameters()).device)
        with Image.open(request.images[0]) as image:
            image_tensor = self._preprocess(image.convert("RGB")).unsqueeze(0).to(device)
        text_tensor = self._tokenizer(candidates).to(device)
        uses_cuda = device.startswith("cuda")
        if uses_cuda:
            self._torch.cuda.reset_peak_memory_stats()
            self._torch.cuda.synchronize()
        started = perf_counter()
        with self._torch.inference_mode():
            image_features = self._model.encode_image(image_tensor)
            text_features = self._model.encode_text(text_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            probabilities = (100 * image_features @ text_features.T).softmax(dim=-1)[0]
        if uses_cuda:
            self._torch.cuda.synchronize()
        duration_ms = (perf_counter() - started) * 1000
        values = probabilities.float().cpu().tolist()
        scores = {candidate: float(score) for candidate, score in zip(candidates, values)}
        return InferenceResult(
            request_id=request.request_id,
            task=TaskType.UNDERSTANDING,
            model_name=self.name,
            text=max(scores, key=scores.get),
            scores=scores,
            metadata={
                "resource": self.name,
                "model_id": self.model_path,
                "model_revision": self.model_revision,
                "model_family": "open_clip",
                "executor": self.recipe.executor.value,
                "device": device,
                "dtype": str(next(self._model.parameters()).dtype).removeprefix("torch."),
                "peak_gpu_memory_mb": (
                    round(self._torch.cuda.max_memory_allocated() / 1024**2, 2)
                    if uses_cuda
                    else None
                ),
                "hostname": platform.node(),
                "scheduler": {
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
                },
                "clinical_use": False,
                **request.metadata,
            },
            duration_ms=round(duration_ms, 2),
        )

    def _medclip_rank(self, request: InferenceRequest) -> InferenceResult:
        if len(request.images) != 1:
            raise ValueError("MedCLIP candidate ranking requires exactly one image.")
        candidates = self._candidates(request)
        with Image.open(request.images[0]) as image:
            inputs = self._processor(
                text=candidates,
                images=image.convert("RGB"),
                return_tensors="pt",
                padding=True,
            )
        inputs = {
            key: value.to(self._device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        started = perf_counter()
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
        duration_ms = (perf_counter() - started) * 1000
        logits = outputs.get("logits_per_image") if isinstance(outputs, dict) else None
        if logits is None and isinstance(outputs, dict):
            logits = outputs.get("logits")
            if logits is None:
                logits = outputs.get("logits_per_text")
            if logits is not None and logits.ndim == 2 and logits.shape[0] == len(candidates):
                logits = logits.transpose(0, 1)
        if logits is None or logits.ndim != 2 or logits.shape[-1] != len(candidates):
            raise RuntimeError("MedCLIP did not return one score for every candidate.")
        probabilities = logits[0].float().softmax(dim=-1).cpu().tolist()
        scores = {candidate: float(score) for candidate, score in zip(candidates, probabilities)}
        return InferenceResult(
            request_id=request.request_id,
            task=TaskType.UNDERSTANDING,
            model_name=self.name,
            text=max(scores, key=scores.get),
            scores=scores,
            metadata={
                "resource": self.name,
                "model_revision": self.model_revision,
                "model_family": "medclip",
                "executor": self.recipe.executor.value,
                "device": self._device,
                "dtype": self._dtype,
                "clinical_use": False,
                **request.metadata,
            },
            duration_ms=round(duration_ms, 2),
        )

    def understand_batch(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        if self._delegate is not None:
            delegated = self._delegate.understand_batch(requests)
            for result in delegated:
                result.model_name = self.name
                result.metadata = {
                    "resource": self.name,
                    "bridge_model": self._delegate.name,
                    **result.metadata,
                }
            return delegated
        if self.recipe.executor in {
            ModelExecutor.QWEN2_VL,
            ModelExecutor.QWEN2_5_VL,
            ModelExecutor.QWEN3_VL,
        }:
            return [self._qwen_generate(request) for request in requests]
        if self.recipe.executor is ModelExecutor.CHEXAGENT:
            return [self._chexagent_generate(request) for request in requests]
        if self.recipe.executor is ModelExecutor.INTERNVL_CHAT:
            return [self._internvl_generate(request) for request in requests]
        if self.recipe.executor is ModelExecutor.M3D_LAMED:
            return [self._m3d_generate(request) for request in requests]
        if self.recipe.executor in {
            ModelExecutor.TRANSFORMERS_PIPELINE,
            ModelExecutor.UNIMED_VL,
        }:
            return [self._hf_generate(request) for request in requests]
        if self.recipe.executor is ModelExecutor.HF_CONTRASTIVE:
            return [self._hf_rank(request) for request in requests]
        if self.recipe.executor is ModelExecutor.OPEN_CLIP_HUB:
            return [self._open_clip_rank(request) for request in requests]
        if self.recipe.executor is ModelExecutor.MEDCLIP:
            return [self._medclip_rank(request) for request in requests]
        raise RuntimeError(f"{self.name} has not been loaded.")

    def close(self) -> None:
        if self._delegate is not None:
            self._delegate.close()
        self._delegate = None
        self._model = None
        self._processor = None
        self._pipeline = None


def catalog_model_factory(resource_name: str):
    def create() -> CatalogModelAdapter:
        return CatalogModelAdapter(resource_name)

    return create
