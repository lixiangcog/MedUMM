from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from medumm.core.io import ensure_directory, write_json
from medumm.training.distributed import DistributedSession
from medumm.training.parallel import unwrap_model


@dataclass(slots=True)
class TrainingState:
    epoch: int = 0
    micro_step: int = 0
    optimizer_step: int = 0
    samples_seen: int = 0


class DistributedCheckpointManager:
    """Save resumable model/optimizer shards plus rank-local runtime state."""

    schema_version = "1.0"

    def __init__(
        self,
        root: str | Path,
        *,
        session: DistributedSession,
        keep_last: int = 3,
        strict_topology: bool = False,
    ) -> None:
        self.root = ensure_directory(Path(root))
        self.session = session
        self.keep_last = keep_last
        self.strict_topology = strict_topology

    def _checkpoint_path(self, step: int) -> Path:
        return self.root / f"step-{step:08d}"

    def latest(self) -> Path | None:
        pointer = self.root / "latest.json"
        if pointer.is_file():
            value = json.loads(pointer.read_text(encoding="utf-8"))
            candidate = self.root / str(value.get("checkpoint", ""))
            if (candidate / "COMPLETED").is_file():
                return candidate
        completed = sorted(
            path for path in self.root.glob("step-*") if (path / "COMPLETED").is_file()
        )
        return completed[-1] if completed else None

    def save(
        self,
        *,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        precision: Any,
        ema: Any,
        state: TrainingState,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        import torch

        path = self._checkpoint_path(state.optimizer_step)
        self.session.barrier()
        if (path / "COMPLETED").is_file():
            return path
        if self.session.is_main_process:
            if path.exists() and not (path / "COMPLETED").exists():
                shutil.rmtree(path)
            ensure_directory(path)
        self.session.barrier()
        distributed_format = self.session.distributed or self.session.config.strategy == "fsdp"
        if distributed_format:
            try:
                import torch.distributed.checkpoint as dcp
                from torch.distributed.checkpoint.state_dict import (
                    StateDictOptions,
                    get_state_dict,
                )
            except (ImportError, ModuleNotFoundError) as error:
                raise RuntimeError(
                    "Sharded checkpointing requires torch.distributed.checkpoint from "
                    "PyTorch 2.2 or newer."
                ) from error
            state_options = StateDictOptions(flatten_optimizer_state_dict=True)
            model_state, optimizer_state = get_state_dict(
                model,
                optimizer,
                options=state_options,
            )
            dcp.save(
                state_dict={"model": model_state, "optimizer": optimizer_state},
                checkpoint_id=str(path / "shards"),
            )
            checkpoint_format = "torch_distributed_checkpoint"
        else:
            torch.save(
                {
                    "model": unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                path / "training.pt",
            )
            checkpoint_format = "torch_save"
        rank_state = {
            "training_state": asdict(state),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "precision": precision.state_dict(),
            "ema": ema.state_dict() if ema is not None else None,
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(rank_state, path / f"rank-{self.session.rank:05d}.pt")
        self.session.barrier()
        if self.session.is_main_process:
            manifest = {
                "schema_version": self.schema_version,
                "format": checkpoint_format,
                "strategy": self.session.config.strategy,
                "world_size": self.session.world_size,
                "state": asdict(state),
                "has_scheduler": scheduler is not None,
                "has_ema": ema is not None,
                "metadata": metadata or {},
            }
            write_json(path / "manifest.json", manifest)
            (path / "COMPLETED").touch()
            write_json(self.root / "latest.json", {"checkpoint": path.name})
            self._prune()
        self.session.barrier()
        return path

    def load(
        self,
        checkpoint: str | Path,
        *,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        precision: Any,
        ema: Any,
    ) -> TrainingState:
        import torch

        path = Path(checkpoint)
        if not path.is_absolute():
            path = self.root / path
        if not (path / "COMPLETED").is_file():
            raise FileNotFoundError(f"Checkpoint is incomplete or missing: {path}")
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        saved_world_size = int(manifest["world_size"])
        if self.strict_topology and saved_world_size != self.session.world_size:
            raise RuntimeError(
                f"Checkpoint world size {saved_world_size} does not match "
                f"current world size {self.session.world_size}."
            )
        if manifest.get("has_ema") and saved_world_size != self.session.world_size:
            raise RuntimeError("EMA shard recovery requires the original world size.")
        if manifest["format"] == "torch_distributed_checkpoint":
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_state_dict,
                set_state_dict,
            )

            state_options = StateDictOptions(flatten_optimizer_state_dict=True)
            model_state, optimizer_state = get_state_dict(
                model,
                optimizer,
                options=state_options,
            )
            payload = {"model": model_state, "optimizer": optimizer_state}
            dcp.load(state_dict=payload, checkpoint_id=str(path / "shards"))
            set_state_dict(
                model,
                optimizer,
                model_state_dict=payload["model"],
                optim_state_dict=payload["optimizer"],
                options=state_options,
            )
        else:
            payload = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
            unwrap_model(model).load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
        sidecar_rank = self.session.rank % saved_world_size
        sidecar = torch.load(
            path / f"rank-{sidecar_rank:05d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        if scheduler is not None and sidecar.get("scheduler") is not None:
            scheduler.load_state_dict(sidecar["scheduler"])
        precision.load_state_dict(sidecar.get("precision", {}))
        if ema is not None and sidecar.get("ema") is not None:
            ema.load_state_dict(sidecar["ema"])
        random.setstate(sidecar["python_rng"])
        torch.set_rng_state(sidecar["torch_rng"])
        if torch.cuda.is_available() and sidecar.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(sidecar["cuda_rng"])
        self.session.barrier()
        return TrainingState(**sidecar["training_state"])

    def _prune(self) -> None:
        completed = sorted(
            path for path in self.root.glob("step-*") if (path / "COMPLETED").is_file()
        )
        for path in completed[: -self.keep_last]:
            shutil.rmtree(path)
