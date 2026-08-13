from __future__ import annotations

import importlib.metadata
import importlib.util
import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


_VERSION_PART = re.compile(r"^(\d+)")


class InferenceBackend(str, Enum):
    NATIVE = "native"
    VLLM = "vllm"
    SGLANG = "sglang"


class BackendMode(str, Enum):
    IN_PROCESS = "in_process"
    OPENAI_HTTP = "openai_http"


@dataclass(frozen=True, slots=True)
class ParallelConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    distributed_executor_backend: str = "mp"

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "ParallelConfig":
        raw = value or {}
        config = cls(
            tensor_parallel_size=int(raw.get("tensor_parallel_size", 1)),
            pipeline_parallel_size=int(raw.get("pipeline_parallel_size", 1)),
            data_parallel_size=int(raw.get("data_parallel_size", 1)),
            distributed_executor_backend=str(
                raw.get("distributed_executor_backend", "mp")
            ),
        )
        if min(
            config.tensor_parallel_size,
            config.pipeline_parallel_size,
            config.data_parallel_size,
        ) < 1:
            raise ValueError("Parallel sizes must all be positive.")
        if config.distributed_executor_backend not in {"mp", "ray", "external_launcher"}:
            raise ValueError(
                "distributed_executor_backend must be mp, ray, or external_launcher."
            )
        return config

    @property
    def world_size(self) -> int:
        return (
            self.tensor_parallel_size
            * self.pipeline_parallel_size
            * self.data_parallel_size
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "world_size": self.world_size}


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    continuous_batching: bool = True
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 8192
    max_queue_size: int = 1024

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "SchedulerConfig":
        raw = value or {}
        config = cls(
            continuous_batching=bool(raw.get("continuous_batching", True)),
            max_num_seqs=int(raw.get("max_num_seqs", 32)),
            max_num_batched_tokens=int(raw.get("max_num_batched_tokens", 8192)),
            max_queue_size=int(raw.get("max_queue_size", 1024)),
        )
        if min(
            config.max_num_seqs,
            config.max_num_batched_tokens,
            config.max_queue_size,
        ) < 1:
            raise ValueError("Scheduler limits must all be positive.")
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BackendConfig:
    name: InferenceBackend
    mode: BackendMode
    parallel: ParallelConfig
    scheduler: SchedulerConfig
    endpoint: str | None = None
    api_key_environment: str | None = None
    request_timeout_seconds: float = 300.0

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "BackendConfig":
        raw = value or {}
        name = InferenceBackend(str(raw.get("name", "native")).casefold())
        default_mode = (
            BackendMode.IN_PROCESS
            if name in {InferenceBackend.NATIVE, InferenceBackend.VLLM}
            else BackendMode.OPENAI_HTTP
        )
        config = cls(
            name=name,
            mode=BackendMode(str(raw.get("mode", default_mode.value)).casefold()),
            parallel=ParallelConfig.from_dict(raw.get("parallel")),
            scheduler=SchedulerConfig.from_dict(raw.get("scheduler")),
            endpoint=str(raw.get("endpoint", "")).strip() or None,
            api_key_environment=(
                str(raw.get("api_key_environment", "")).strip() or None
            ),
            request_timeout_seconds=float(raw.get("request_timeout_seconds", 300.0)),
        )
        if config.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive.")
        if config.name is InferenceBackend.NATIVE:
            if config.mode is not BackendMode.IN_PROCESS:
                raise ValueError("The native backend is in-process only.")
            if config.parallel.world_size != 1:
                raise ValueError("The native backend does not provide model parallelism.")
        if config.name is InferenceBackend.SGLANG:
            if config.mode is not BackendMode.OPENAI_HTTP:
                raise ValueError("The SGLang backend is supported through OpenAI HTTP mode.")
            if not config.endpoint:
                raise ValueError("The SGLang HTTP backend requires backend.endpoint.")
        if config.name is InferenceBackend.VLLM and config.mode is BackendMode.OPENAI_HTTP:
            if not config.endpoint:
                raise ValueError("The vLLM HTTP backend requires backend.endpoint.")
        return config

    @property
    def continuous_batching(self) -> bool:
        return (
            self.scheduler.continuous_batching
            and self.name in {InferenceBackend.VLLM, InferenceBackend.SGLANG}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "mode": self.mode.value,
            "parallel": self.parallel.to_dict(),
            "scheduler": self.scheduler.to_dict(),
            "endpoint": self.endpoint,
            "api_key_environment": self.api_key_environment,
            "request_timeout_seconds": self.request_timeout_seconds,
            "continuous_batching": self.continuous_batching,
        }


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    name: InferenceBackend
    installed: bool
    version: str | None
    modes: tuple[BackendMode, ...]
    continuous_batching: bool
    tensor_parallel: bool
    pipeline_parallel: bool
    data_parallel: bool
    openai_compatible: bool
    emu3_5_native_cfg: bool
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["name"] = self.name.value
        value["modes"] = [mode.value for mode in self.modes]
        value["notes"] = list(self.notes)
        return value


def _installed_version(distribution: str, module: str | None = None) -> str | None:
    if importlib.util.find_spec(module or distribution.replace("-", "_")) is None:
        return None
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def backend_catalog() -> list[dict[str, Any]]:
    vllm_version = _installed_version("vllm")
    sglang_version = _installed_version("sglang")
    emu3_5_scheduler_installed = (
        vllm_version == "0.11.0"
        and importlib.util.find_spec("vllm.v1.core.sched.batch_scheduler") is not None
    )
    values = (
        BackendCapabilities(
            name=InferenceBackend.NATIVE,
            installed=True,
            version=None,
            modes=(BackendMode.IN_PROCESS,),
            continuous_batching=False,
            tensor_parallel=False,
            pipeline_parallel=False,
            data_parallel=False,
            openai_compatible=False,
            emu3_5_native_cfg=False,
            notes=("Model adapter controls batching and device placement.",),
        ),
        BackendCapabilities(
            name=InferenceBackend.VLLM,
            installed=vllm_version is not None,
            version=vllm_version,
            modes=(BackendMode.IN_PROCESS, BackendMode.OPENAI_HTTP),
            continuous_batching=True,
            tensor_parallel=True,
            pipeline_parallel=True,
            data_parallel=True,
            openai_compatible=True,
            emu3_5_native_cfg=emu3_5_scheduler_installed,
            notes=(
                "Emu3.5 native CFG additionally requires BAAI's vLLM 0.11.0 patches.",
            ),
        ),
        BackendCapabilities(
            name=InferenceBackend.SGLANG,
            installed=sglang_version is not None,
            version=sglang_version,
            modes=(BackendMode.OPENAI_HTTP,),
            continuous_batching=True,
            tensor_parallel=True,
            pipeline_parallel=True,
            data_parallel=True,
            openai_compatible=True,
            emu3_5_native_cfg=False,
            notes=(
                "No upstream Emu3.5 cond/uncond CFG scheduler is declared; use vLLM for Emu3.5 generation.",
            ),
        ),
    )
    return [value.to_dict() for value in values]


def require_backend(config: BackendConfig) -> dict[str, Any]:
    capabilities = {item["name"]: item for item in backend_catalog()}[config.name.value]
    if config.name is not InferenceBackend.NATIVE and config.mode is BackendMode.IN_PROCESS:
        if not capabilities["installed"]:
            raise RuntimeError(
                f"Inference backend {config.name.value!r} is not installed in this runtime."
            )
    return capabilities


def version_at_least(version: str, minimum: tuple[int, ...]) -> bool:
    parts: list[int] = []
    for raw in re.split(r"[._+-]", version):
        match = _VERSION_PART.match(raw)
        if match is None:
            break
        parts.append(int(match.group(1)))
    padded = tuple(parts + [0] * max(0, len(minimum) - len(parts)))
    return padded[: len(minimum)] >= minimum
