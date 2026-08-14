from __future__ import annotations

import random
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from medumm.core.io import ensure_directory
from medumm.training.checkpoint import DistributedCheckpointManager, TrainingState
from medumm.training.config import DistributedTrainingConfig
from medumm.training.distributed import DistributedSession, distributed_runtime_metadata
from medumm.training.ema import ExponentialMovingAverage
from medumm.training.parallel import clip_grad_norm, wrap_model
from medumm.training.precision import PrecisionManager


StepFunction = Callable[[Any, Any], Any]


def move_to_device(value: Any, device: Any) -> Any:
    if hasattr(value, "to") and callable(value.to):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(child, device) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(child, device) for child in value)
    if isinstance(value, list):
        return [move_to_device(child, device) for child in value]
    return value


def create_dataloader(
    dataset: Any,
    *,
    session: DistributedSession,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    drop_last: bool = False,
    collate_fn: Any = None,
) -> Any:
    import torch

    sampler = None
    if session.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=session.world_size,
            rank=session.rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn,
        generator=generator,
    )


class DistributedTrainingEngine:
    """Generic training loop shared by model-specific MedUMM post-trainers."""

    def __init__(
        self,
        model: Any,
        *,
        session: DistributedSession,
        config: DistributedTrainingConfig,
        output_directory: str | Path,
        optimizer_factory: Callable[[Any], Any],
        scheduler_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        import torch

        self.torch = torch
        self.session = session
        self.config = config
        self.output_directory = ensure_directory(Path(output_directory))
        self.model = wrap_model(model, session=session, config=config)
        self.optimizer = optimizer_factory(self.model.parameters())
        self.scheduler = scheduler_factory(self.optimizer) if scheduler_factory else None
        self.precision = PrecisionManager(
            torch,
            precision=config.precision,
            device_type=session.device.type,
        )
        self.ema = (
            ExponentialMovingAverage(
                self.model,
                decay=config.ema_decay,
                device=config.ema_device,
            )
            if config.ema_decay is not None
            else None
        )
        self.checkpoints = DistributedCheckpointManager(
            self.output_directory / "checkpoints",
            session=session,
            keep_last=config.checkpoint_keep_last,
            strict_topology=config.checkpoint_strict_topology,
        )
        self.state = TrainingState()
        self.history_path = self.output_directory / "history.jsonl"

    def resume(self, value: str | Path | None) -> Path | None:
        if value is None or str(value).casefold() in {"", "none", "false"}:
            return None
        checkpoint = self.checkpoints.latest() if str(value).casefold() == "auto" else Path(value)
        if checkpoint is None:
            return None
        self.state = self.checkpoints.load(
            checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            precision=self.precision,
            ema=self.ema,
        )
        return checkpoint

    def fit(
        self,
        dataloader: Any,
        *,
        step_function: StepFunction,
        epochs: int,
        resume_from: str | Path | None = None,
        max_optimizer_steps: int | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if epochs < 1:
            raise ValueError("epochs must be positive.")
        try:
            loader_length = len(dataloader)
        except TypeError as error:
            raise ValueError("DistributedTrainingEngine requires a sized dataloader.") from error
        if loader_length < 1:
            raise ValueError("Training dataloader is empty.")
        resumed = self.resume(resume_from)
        seed = int((checkpoint_metadata or {}).get("seed", 42))
        if resumed is None:
            random.seed(seed + self.session.rank)
            self.torch.manual_seed(seed + self.session.rank)
            if self.torch.cuda.is_available():
                self.torch.cuda.manual_seed_all(seed + self.session.rank)
        started = perf_counter()
        loss_sum = 0.0
        loss_count = 0
        stopped_early = False
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(self.state.epoch, epochs):
            sampler = getattr(dataloader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            self.model.train()
            skip_before = self.state.micro_step if epoch == self.state.epoch else 0
            for batch_index, batch in enumerate(dataloader):
                if batch_index < skip_before:
                    continue
                window_start = (batch_index // self.config.gradient_accumulation_steps) * self.config.gradient_accumulation_steps
                window_size = min(
                    self.config.gradient_accumulation_steps,
                    loader_length - window_start,
                )
                should_step = (
                    (batch_index + 1) % self.config.gradient_accumulation_steps == 0
                    or batch_index + 1 == loader_length
                )
                sync_context = (
                    nullcontext()
                    if should_step or not hasattr(self.model, "no_sync")
                    else self.model.no_sync()
                )
                batch = move_to_device(batch, self.session.device)
                with sync_context:
                    with self.precision.autocast():
                        output = step_function(self.model, batch)
                        loss = output[0] if isinstance(output, tuple) else output
                        scaled_loss = loss / window_size
                    if not bool(self.torch.isfinite(loss).all()):
                        raise FloatingPointError(
                            f"Non-finite loss at epoch {epoch}, batch {batch_index}."
                        )
                    self.precision.backward(scaled_loss)
                loss_sum += float(loss.detach().float().item())
                loss_count += 1
                self.state.micro_step = batch_index + 1
                self.state.samples_seen += self._batch_size(batch) * self.session.world_size
                if not should_step:
                    continue
                self.precision.unscale_(self.optimizer)
                gradient_norm = None
                if self.config.max_grad_norm is not None:
                    gradient_norm = clip_grad_norm(self.model, self.config.max_grad_norm)
                self.precision.step(self.optimizer)
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()
                if self.ema is not None:
                    self.ema.update(self.model)
                self.state.optimizer_step += 1
                self._write_history(
                    {
                        "epoch": epoch,
                        "micro_step": batch_index + 1,
                        "optimizer_step": self.state.optimizer_step,
                        "loss": float(loss.detach().float().item()),
                        "gradient_norm": gradient_norm,
                        "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                    }
                )
                if (
                    self.config.checkpoint_every_steps
                    and self.state.optimizer_step % self.config.checkpoint_every_steps == 0
                ):
                    self._save(checkpoint_metadata)
                if max_optimizer_steps is not None and self.state.optimizer_step >= max_optimizer_steps:
                    stopped_early = True
                    break
            if stopped_early:
                break
            self.state.epoch = epoch + 1
            self.state.micro_step = 0
        checkpoint = self._save(checkpoint_metadata)
        mean_loss = self.session.reduce_mean(loss_sum / max(loss_count, 1))
        return {
            "status": "interrupted" if stopped_early else "completed",
            "checkpoint": str(checkpoint),
            "resumed_from": str(resumed) if resumed else None,
            "state": asdict(self.state),
            "mean_loss": mean_loss,
            "duration_seconds": round(perf_counter() - started, 3),
            "distributed": distributed_runtime_metadata(self.session),
            "precision": self.config.precision,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "ema_updates": self.ema.num_updates if self.ema is not None else 0,
            "activation_checkpointing": self.config.activation_checkpointing,
        }

    def _save(self, metadata: dict[str, Any] | None) -> Path:
        return self.checkpoints.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            precision=self.precision,
            ema=self.ema,
            state=self.state,
            metadata=metadata,
        )

    def _write_history(self, row: dict[str, Any]) -> None:
        if self.session.is_main_process:
            import json

            with self.history_path.open("a", encoding="utf-8") as writer:
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _batch_size(batch: Any) -> int:
        values = batch.values() if isinstance(batch, dict) else batch
        if not isinstance(values, (list, tuple)):
            values = list(values) if not hasattr(values, "shape") else [values]
        for value in values:
            shape = getattr(value, "shape", ())
            if shape:
                return int(shape[0])
        return 1
