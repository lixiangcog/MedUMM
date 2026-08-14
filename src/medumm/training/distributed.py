from __future__ import annotations

import os
from collections.abc import MutableMapping
from contextlib import AbstractContextManager
from datetime import timedelta
from typing import Any

from medumm.core.distributed import DistributedContext
from medumm.training.config import DistributedTrainingConfig


def normalize_distributed_environment(
    environment: MutableMapping[str, str] | None = None,
) -> DistributedContext:
    """Expose Slurm ranks through the environment contract used by torchrun."""

    values = os.environ if environment is None else environment
    mappings = {
        "RANK": "SLURM_PROCID",
        "LOCAL_RANK": "SLURM_LOCALID",
        "WORLD_SIZE": "SLURM_NTASKS",
    }
    for target, source in mappings.items():
        if target not in values and source in values:
            values[target] = values[source]
    return DistributedContext(
        rank=int(values.get("RANK", values.get("SLURM_PROCID", "0"))),
        local_rank=int(values.get("LOCAL_RANK", values.get("SLURM_LOCALID", "0"))),
        world_size=int(values.get("WORLD_SIZE", values.get("SLURM_NTASKS", "1"))),
    )


class DistributedSession(AbstractContextManager["DistributedSession"]):
    """Own process-group initialization, device binding, collectives, and teardown."""

    def __init__(
        self,
        config: DistributedTrainingConfig,
        *,
        requested_device: str = "auto",
    ) -> None:
        self.config = config
        self.context = normalize_distributed_environment()
        self.requested_device = requested_device
        self._owns_process_group = False
        self._torch: Any = None
        self.device: Any = None
        self.backend: str | None = None

    @property
    def rank(self) -> int:
        return self.context.rank

    @property
    def local_rank(self) -> int:
        return self.context.local_rank

    @property
    def world_size(self) -> int:
        return self.context.world_size

    @property
    def is_main_process(self) -> bool:
        return self.context.is_main_process

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    def __enter__(self) -> "DistributedSession":
        try:
            import torch
            import torch.distributed as dist
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Distributed training requires PyTorch; install MedUMM with the "
                "distributed extra."
            ) from error
        self._torch = torch
        requested = self.requested_device.casefold()
        use_cuda = torch.cuda.is_available() and requested not in {"cpu", "mps"}
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA training was requested but CUDA is unavailable.")
        if use_cuda:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu" if requested == "auto" else requested)
        self.backend = (
            "nccl" if self.device.type == "cuda" else "gloo"
        ) if self.config.backend == "auto" else self.config.backend
        needs_group = self.distributed or self.config.strategy == "fsdp"
        if needs_group:
            if not dist.is_available():
                raise RuntimeError("The installed PyTorch build has no distributed support.")
            if not dist.is_initialized():
                if self.config.init_method == "env://":
                    missing = [
                        name
                        for name in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
                        if name not in os.environ
                    ]
                    if missing:
                        raise RuntimeError(
                            "Distributed initialization is missing environment variables: "
                            + ", ".join(missing)
                        )
                kwargs: dict[str, Any] = {
                    "backend": self.backend,
                    "init_method": self.config.init_method,
                    "rank": self.rank,
                    "world_size": self.world_size,
                    "timeout": timedelta(seconds=self.config.timeout_seconds),
                }
                dist.init_process_group(**kwargs)
                self._owns_process_group = True
            elif dist.get_rank() != self.rank or dist.get_world_size() != self.world_size:
                raise RuntimeError("Existing process group does not match the launch environment.")
        return self

    def barrier(self) -> None:
        if self._torch is None:
            raise RuntimeError("DistributedSession has not been entered.")
        dist = self._torch.distributed
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def reduce_mean(self, value: float) -> float:
        if not self.distributed:
            return float(value)
        tensor = self._torch.tensor(float(value), device=self.device, dtype=self._torch.float64)
        self._torch.distributed.all_reduce(tensor)
        return float((tensor / self.world_size).item())

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._torch is None:
            return None
        dist = self._torch.distributed
        if self._owns_process_group and dist.is_available() and dist.is_initialized():
            try:
                if exc_type is None:
                    dist.barrier()
            finally:
                dist.destroy_process_group()
        return None


def distributed_runtime_metadata(session: DistributedSession) -> dict[str, Any]:
    return {
        "strategy": session.config.strategy,
        "backend": session.backend,
        "rank": session.rank,
        "local_rank": session.local_rank,
        "world_size": session.world_size,
        "device": str(session.device),
    }
