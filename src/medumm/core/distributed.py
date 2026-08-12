from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar


Item = TypeVar("Item")


@dataclass(frozen=True, slots=True)
class DistributedContext:
    """Dependency-light view of a torchrun or Slurm process group."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @classmethod
    def from_environment(cls) -> "DistributedContext":
        return cls(
            rank=int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0"))),
            local_rank=int(
                os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))
            ),
            world_size=int(
                os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1"))
            ),
        )

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def shard(self, values: Sequence[Item]) -> list[Item]:
        return list(values[self.rank :: self.world_size])

    def barrier(self) -> None:
        if not self.enabled:
            return
        try:
            import torch.distributed as distributed
        except ModuleNotFoundError as error:
            raise RuntimeError("Distributed execution requires PyTorch.") from error
        if not distributed.is_available() or not distributed.is_initialized():
            raise RuntimeError("A distributed process group has not been initialized.")
        distributed.barrier()
