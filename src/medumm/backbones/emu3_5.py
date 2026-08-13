from __future__ import annotations

import importlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from medumm.core.contracts import ArchitectureFamily, Modality, ModelCapabilities, TaskType
from medumm.core.interfaces import ModelAdapter
from medumm.core.results import Artifact, InferenceResult
from medumm.core.runtime import RuntimeContext
from medumm.inference.backends import (
    BackendConfig,
    BackendMode,
    InferenceBackend,
    require_backend,
)
from medumm.inference.request import InferenceRequest


OFFICIAL_REPOSITORY = "https://github.com/baaivision/Emu3.5"
REQUIRED_VLLM_VERSION = "0.11.0"
PATCHED_ARCHITECTURE = "Emu3_5ForCausalLM"
PATCHED_SCHEDULER = "vllm.v1.core.sched.batch_scheduler.Scheduler"
_MUTABLE_REVISIONS = frozenset({"", "main", "master", "latest", "head"})
_SPECIAL_TOKENS = {
    "BOS": "<|extra_203|>",
    "EOS": "<|extra_204|>",
    "EOL": "<|extra_200|>",
    "EOI": "<|image end|>",
    "SOI": "<|image token|>",
}
_DEFAULT_SAMPLING = {
    "text_top_k": 1024,
    "text_top_p": 0.9,
    "text_temperature": 1.0,
    "image_top_k": 5120,
    "image_top_p": 1.0,
    "image_temperature": 1.0,
    "top_k": 131072,
    "top_p": 1.0,
    "temperature": 1.0,
    "max_new_tokens": 5120,
    "do_sample": True,
}


def _import_from_emu(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            f"The pinned Emu3.5 source cannot import {module_name!r}."
        ) from error


def _template(task: str, with_image: bool) -> tuple[str, str]:
    if with_image:
        unc = (
            "<|extra_203|>You are a helpful assistant. USER: <|IMAGE|> "
            "ASSISTANT: <|extra_100|>"
        )
        prompt = (
            f"<|extra_203|>You are a helpful assistant for {task} task. USER: "
            "{question}<|IMAGE|> ASSISTANT: <|extra_100|>"
        )
    else:
        unc = (
            "<|extra_203|>You are a helpful assistant. USER:  "
            "ASSISTANT: <|extra_100|>"
        )
        prompt = (
            f"<|extra_203|>You are a helpful assistant for {task} task. USER: "
            "{question} ASSISTANT: <|extra_100|>"
        )
    return unc, prompt


class Emu3_5Adapter(ModelAdapter):
    """Native Emu3.5 adapter for BAAI's patched vLLM runtime.

    The implementation intentionally refuses ordinary vLLM and SGLang for
    Emu3.5 CFG. Its cond/uncond request pairing depends on upstream patches
    that add both the model architecture and a custom scheduler.
    """

    name = "emu3_5"
    capabilities = ModelCapabilities(
        tasks=frozenset({TaskType.UNDERSTANDING, TaskType.GENERATION, TaskType.EDITING}),
        input_modalities=frozenset({Modality.TEXT, Modality.IMAGE, Modality.IMAGE_SET}),
        output_modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
        architecture=ArchitectureFamily.AUTOREGRESSIVE,
        supports_batching=True,
        max_batch_size=None,
        max_images=4,
        supports_hidden_states=False,
        supported_backends=frozenset({"vllm"}),
        supports_continuous_batching=True,
        supports_classifier_free_guidance=True,
        parallelism=frozenset({"tensor_parallel"}),
        notes=(
            "Requires vLLM 0.11.0 plus BAAI's official Emu3.5 patches.",
            "SGLang is not accepted for Emu3.5 CFG because no equivalent upstream scheduler is declared.",
            "Research use only; generated medical imagery is synthetic and not clinical evidence.",
        ),
    )

    def load(self, config: dict[str, Any], runtime: RuntimeContext) -> None:
        self.runtime = runtime
        self.backend = BackendConfig.from_dict(
            config.get("backend", {"name": "vllm", "mode": "in_process"})
        )
        if self.backend.name is InferenceBackend.SGLANG:
            raise RuntimeError(
                "SGLang does not provide the official Emu3.5 cond/uncond CFG scheduler; "
                "use backend.name=vllm with BAAI's patches."
            )
        if self.backend.name is not InferenceBackend.VLLM:
            raise RuntimeError("Emu3.5 currently requires backend.name=vllm.")
        if self.backend.mode is not BackendMode.IN_PROCESS:
            raise RuntimeError(
                "Emu3.5 CFG requires the patched in-process vLLM API; the standard "
                "OpenAI endpoint does not expose uncond_prompt_token_ids."
            )
        capabilities = require_backend(self.backend)
        if capabilities["version"] != REQUIRED_VLLM_VERSION:
            raise RuntimeError(
                f"Emu3.5 requires vLLM {REQUIRED_VLLM_VERSION}; found "
                f"{capabilities['version']!r}."
            )
        if self.backend.parallel.pipeline_parallel_size != 1:
            raise ValueError("The patched Emu3.5 vLLM 0.11.0 path supports tensor parallel only.")
        if self.backend.parallel.data_parallel_size != 1:
            raise ValueError(
                "Use independent Emu3.5 replicas for data parallelism; the in-process "
                "adapter accepts data_parallel_size=1 only."
            )

        self.source_revision = str(config.get("source_revision", "")).strip()
        if self.source_revision.casefold() in _MUTABLE_REVISIONS:
            raise ValueError("Emu3.5 requires an immutable source_revision.")
        self.model_revision = str(config.get("model_revision", "")).strip()
        if self.model_revision.casefold() in _MUTABLE_REVISIONS:
            raise ValueError("Emu3.5 requires an immutable model_revision.")
        self.vq_revision = str(config.get("vq_revision", "")).strip()
        if self.vq_revision.casefold() in _MUTABLE_REVISIONS:
            raise ValueError("Emu3.5 requires an immutable vq_revision.")

        self.source_root = Path(str(config.get("source_root", ""))).expanduser()
        if not self.source_root.is_absolute():
            self.source_root = runtime.project_root / self.source_root
        if not (self.source_root / "src/utils/model_utils.py").is_file():
            raise FileNotFoundError(
                f"source_root is not a pinned Emu3.5 checkout: {self.source_root}"
            )
        self._verify_source_checkout()
        self.model_path = str(config.get("model_path", "BAAI/Emu3.5")).strip()
        self.vq_path = str(config.get("vq_path", "BAAI/Emu3.5-VisionTokenizer")).strip()
        self.tokenizer_path = str(
            config.get("tokenizer_path", self.source_root / "src/tokenizer_emu3_ibq")
        )
        self.vq_device = str(config.get("vq_device", "cuda:0"))
        self.gpu_memory_utilization = float(config.get("gpu_memory_utilization", 0.7))
        self.seed = int(config.get("seed", 6666))
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1].")
        configured = config.get("generation", {}) or {}
        if not isinstance(configured, dict):
            raise ValueError("Emu3.5 generation config must be a mapping.")
        self.defaults = {
            **_DEFAULT_SAMPLING,
            "classifier_free_guidance": 5.0,
            "image_area": 1048576,
            **configured,
        }
        self.output_directory = Path(
            str(config.get("output_directory", "outputs/inference/emu3_5"))
        )
        if not self.output_directory.is_absolute():
            self.output_directory = runtime.project_root / self.output_directory

        source_string = str(self.source_root.resolve())
        if source_string not in sys.path:
            sys.path.insert(0, source_string)
        _import_from_emu("src.utils.model_utils")
        self.model, self.tokenizer, self.vq_model = self._build_engine()
        self._verify_patched_engine()
        self.special_token_ids = {
            name: self.tokenizer.encode(token)[0] for name, token in _SPECIAL_TOKENS.items()
        }

    def _verify_source_checkout(self) -> None:
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.source_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise RuntimeError(
                "Emu3.5 source_root must retain Git metadata so source_revision "
                "can be verified."
            ) from error
        actual = completed.stdout.strip().casefold()
        expected = self.source_revision.casefold()
        if actual != expected:
            raise RuntimeError(
                f"Emu3.5 source revision mismatch: expected {expected!r}, "
                f"found {actual!r}."
            )

    def _build_engine(self) -> tuple[Any, Any, Any]:
        """Build the official components while exposing scheduler controls.

        The official builder hard-codes its queue limits. Reconstructing the
        small composition step lets MedUMM honor its public scheduler contract
        without copying or modifying upstream model implementations.
        """

        from transformers import AutoTokenizer
        from vllm import LLM

        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            special_tokens_file=str(Path(self.tokenizer_path) / "emu3_vision_tokens.txt"),
            trust_remote_code=True,
        )
        tokenizer.bos_token = "<|extra_203|>"
        tokenizer.eos_token = "<|extra_204|>"
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.eol_token = "<|extra_200|>"
        tokenizer.eoi_token = "<|image end|>"
        tokenizer.img_token = "<|image token|>"
        vision_tokenizer = _import_from_emu("src.vision_tokenizer")
        vq_model = vision_tokenizer.build_vision_tokenizer(
            "ibq", self.vq_path, device=self.vq_device
        )
        resolution_map = {
            tokenizer.encode(digit)[0]: digit for digit in "0123456789*"
        }
        model = LLM(
            self.model_path,
            revision=self.model_revision,
            tokenizer=self.tokenizer_path,
            trust_remote_code=True,
            dtype="auto",
            tensor_parallel_size=self.backend.parallel.tensor_parallel_size,
            distributed_executor_backend=self.backend.parallel.distributed_executor_backend,
            gpu_memory_utilization=self.gpu_memory_utilization,
            disable_log_stats=False,
            enable_chunked_prefill=False,
            enable_prefix_caching=False,
            max_num_batched_tokens=self.backend.scheduler.max_num_batched_tokens,
            max_num_seqs=self.backend.scheduler.max_num_seqs,
            seed=self.seed,
            generation_config="vllm",
            scheduler_cls=PATCHED_SCHEDULER,
            compilation_config={
                "full_cuda_graph": True,
                "backend": "cudagraph",
                "cudagraph_capture_sizes": [
                    1,
                    min(2, self.backend.scheduler.max_num_seqs),
                ],
            },
            additional_config={
                "boi_token_id": tokenizer.encode("<|image start|>")[0],
                "soi_token_id": tokenizer.encode("<|image token|>")[0],
                "eol_token_id": tokenizer.encode("<|extra_200|>")[0],
                "eoi_token_id": tokenizer.encode("<|image end|>")[0],
                "resolution_map": resolution_map,
            },
        )
        model.set_tokenizer(tokenizer)
        return model, tokenizer, vq_model

    def _verify_patched_engine(self) -> None:
        if importlib.util.find_spec("vllm.v1.core.sched.batch_scheduler") is None:
            raise RuntimeError(
                "BAAI's Emu3.5 vLLM patches are missing: batch_scheduler is unavailable."
            )
        config = getattr(getattr(self.model, "llm_engine", None), "vllm_config", None)
        model_config = getattr(config, "model_config", None)
        architectures = tuple(getattr(model_config, "architectures", ()) or ())
        if PATCHED_ARCHITECTURE not in architectures:
            raise RuntimeError(
                f"BAAI's patched architecture {PATCHED_ARCHITECTURE!r} is not active: "
                f"{architectures!r}."
            )
        scheduler_config = getattr(config, "scheduler_config", None)
        scheduler = str(getattr(scheduler_config, "scheduler_cls", ""))
        if "batch_scheduler" not in scheduler:
            raise RuntimeError("BAAI's Emu3.5 CFG batch scheduler is not active.")

    def _encoded(self, request: InferenceRequest) -> tuple[Any, Any, str]:
        task = "understanding" if request.task is TaskType.UNDERSTANDING else (
            "x2i" if request.task is TaskType.EDITING else "t2i"
        )
        with_image = bool(request.images)
        unc_prompt, template = _template(task, with_image)
        prompt = template.format(question=str(request.prompt or ""))
        if with_image:
            input_utils = _import_from_emu("src.utils.input_utils")
            cfg = SimpleNamespace(image_area=int(request.parameters.get(
                "image_area", self.defaults["image_area"]
            )))
            images = request.images[-4:]
            encoded: list[str] = []
            for path in images:
                with Image.open(path) as source:
                    encoded.append(
                        input_utils.build_image(
                            source.convert("RGB"), cfg, self.tokenizer, self.vq_model
                        )
                    )
            encoded_images = "".join(encoded)
            prompt = prompt.replace("<|IMAGE|>", encoded_images)
            unc_prompt = unc_prompt.replace("<|IMAGE|>", encoded_images)
        input_ids = self.tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=False
        )
        if int(input_ids[0, 0]) != self.special_token_ids["BOS"]:
            import torch

            bos = torch.tensor([[self.special_token_ids["BOS"]]], dtype=input_ids.dtype)
            input_ids = torch.cat([bos, input_ids], dim=1)
        unconditional_ids = self.tokenizer.encode(
            unc_prompt, return_tensors="pt", add_special_tokens=False
        )
        return input_ids, unconditional_ids, task

    def _sampling_params(self, request: InferenceRequest, task: str) -> Any:
        from vllm import SamplingParams

        options = {**self.defaults, **request.parameters}
        sampling = {**_DEFAULT_SAMPLING, **options}
        guidance_scale = float(sampling["classifier_free_guidance"])
        if guidance_scale < 1:
            raise ValueError("classifier_free_guidance must be at least 1.0.")
        if task == "understanding" and not bool(sampling.get("do_sample", False)):
            sampling.update(
                {"top_k": 1, "text_top_k": 1, "temperature": 1.0, "text_temperature": 1.0}
            )
        extra_args = {
            "guidance_scale": guidance_scale,
            "text_top_k": int(sampling["text_top_k"]),
            "text_top_p": float(sampling["text_top_p"]),
            "text_temperature": float(sampling["text_temperature"]),
            "visual_top_k": int(sampling["image_top_k"]),
            "visual_top_p": float(sampling["image_top_p"]),
            "visual_temperature": float(sampling["image_temperature"]),
            "area": int(sampling["image_area"]),
        }
        stop = self.special_token_ids["EOI"] if task in {"t2i", "x2i"} else self.special_token_ids["EOS"]
        return SamplingParams(
            top_k=int(sampling["top_k"]),
            top_p=float(sampling["top_p"]),
            temperature=float(sampling["temperature"]),
            max_tokens=int(sampling["max_new_tokens"]),
            detokenize=False,
            extra_args=extra_args,
            stop_token_ids=[stop],
        )

    def _prepare(self, request: InferenceRequest) -> tuple[dict[str, Any], Any, str]:
        input_ids, unconditional_ids, task = self._encoded(request)
        return (
            {
                "prompt_token_ids": input_ids.tolist()[0],
                "uncond_prompt_token_ids": unconditional_ids.tolist()[0],
            },
            self._sampling_params(request, task),
            task,
        )

    def _decode(self, request: InferenceRequest, output: Any, task: str) -> InferenceResult:
        token_ids = output.outputs[0].token_ids
        decoded = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        generation_utils = _import_from_emu("src.utils.generation_utils")
        multimodal = generation_utils.multimodal_decode(decoded, self.tokenizer, self.vq_model)
        text_parts: list[str] = []
        artifacts: list[Artifact] = []
        output_base = (
            Path(request.output_path)
            if request.output_path
            else self.output_directory / f"{request.request_id}.png"
        )
        for index, (kind, payload) in enumerate(multimodal):
            if kind == "text" and isinstance(payload, str):
                text_parts.append(payload)
            elif kind == "image" and isinstance(payload, Image.Image):
                path = output_base
                if index:
                    path = path.with_name(f"{path.stem}-{index}{path.suffix or '.png'}")
                path.parent.mkdir(parents=True, exist_ok=True)
                payload.save(path, format="PNG")
                artifacts.append(Artifact("image", str(path), "image/png"))
        metrics = getattr(output, "metrics", None)
        finished = getattr(metrics, "finished_time", None)
        first = getattr(metrics, "first_token_time", None)
        arrival = getattr(metrics, "arrival_time", None)
        scheduled = getattr(metrics, "scheduled_time", None)
        return InferenceResult(
            request_id=request.request_id,
            task=request.task,
            model_name=self.name,
            text="".join(text_parts).strip() or None,
            artifacts=artifacts,
            metadata={
                "backend": self.backend.to_dict(),
                "cfg_scheduler": PATCHED_SCHEDULER,
                "classifier_free_guidance": float(
                    request.parameters.get(
                        "classifier_free_guidance", self.defaults["classifier_free_guidance"]
                    )
                ),
                "generated_tokens": len(token_ids),
                "queue_ms": round((scheduled - arrival) * 1000, 3)
                if scheduled is not None and arrival is not None
                else None,
                "time_to_first_token_ms": round((first - arrival) * 1000, 3)
                if first is not None and arrival is not None
                else None,
                "engine_latency_ms": round((finished - arrival) * 1000, 3)
                if finished is not None and arrival is not None
                else None,
                "source_repository": OFFICIAL_REPOSITORY,
                "source_revision": self.source_revision,
                "model_revision": self.model_revision,
                "vq_revision": self.vq_revision,
                "hostname": platform.node(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "clinical_use": False,
                **request.metadata,
            },
        )

    def _run_batch(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        prepared = [self._prepare(request) for request in requests]
        inputs = [item[0] for item in prepared]
        sampling = [item[1] for item in prepared]
        started = time.perf_counter()
        outputs = self.model.generate(inputs, sampling_params=sampling, use_tqdm=False)
        wall_ms = (time.perf_counter() - started) * 1000
        by_id = {str(output.request_id): output for output in outputs}
        results: list[InferenceResult] = []
        for index, (request, item) in enumerate(zip(requests, prepared, strict=True)):
            output = by_id.get(str(index), outputs[index])
            result = self._decode(request, output, item[2])
            result.duration_ms = round(wall_ms, 3)
            result.metadata["continuous_batch_size"] = len(requests)
            results.append(result)
        return results

    def understand_batch(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        return self._run_batch(requests)

    def generate_batch(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        return self._run_batch(requests)

    def edit_batch(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        return self._run_batch(requests)

    def runtime_info(self) -> dict[str, Any]:
        return {
            "backend": self.backend.to_dict(),
            "model_path": self.model_path,
            "model_revision": self.model_revision,
            "vq_path": self.vq_path,
            "vq_revision": self.vq_revision,
            "source_root": str(self.source_root),
            "source_revision": self.source_revision,
            "vllm_version": REQUIRED_VLLM_VERSION,
            "cfg_scheduler": PATCHED_SCHEDULER,
            "patched_architecture": PATCHED_ARCHITECTURE,
        }

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        if importlib.util.find_spec("torch") is not None:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
