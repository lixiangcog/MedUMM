from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


_STRATEGIES = {"single", "ddp", "fsdp"}
_PRECISIONS = {"fp32", "fp16", "bf16"}
_SHARDING_STRATEGIES = {
    "full_shard",
    "shard_grad_op",
    "hybrid_shard",
    "no_shard",
}


@dataclass(frozen=True, slots=True)
class DistributedTrainingConfig:
    """Stable, model-independent configuration for distributed post-training."""

    strategy: str = "single"
    backend: str = "auto"
    init_method: str = "env://"
    timeout_seconds: int = 600
    precision: str = "fp32"
    gradient_accumulation_steps: int = 1
    max_grad_norm: float | None = 1.0
    find_unused_parameters: bool = False
    static_graph: bool = False
    activation_checkpointing: bool = False
    activation_checkpoint_module_classes: tuple[str, ...] = ()
    activation_checkpoint_min_params: int = 0
    fsdp_sharding_strategy: str = "full_shard"
    fsdp_min_num_params: int = 100_000_000
    fsdp_cpu_offload: bool = False
    fsdp_sync_module_states: bool = True
    fsdp_use_orig_params: bool = True
    ema_decay: float | None = None
    ema_device: str = "model"
    checkpoint_every_steps: int = 0
    checkpoint_keep_last: int = 3
    checkpoint_strict_topology: bool = False

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any] | None,
        *,
        world_size: int = 1,
    ) -> "DistributedTrainingConfig":
        raw = dict(value or {})
        strategy = str(raw.get("strategy", "ddp" if world_size > 1 else "single")).casefold()
        precision = str(raw.get("precision", "fp32")).casefold()
        sharding = str(raw.get("fsdp_sharding_strategy", "full_shard")).casefold()
        max_grad_norm = raw.get("max_grad_norm", 1.0)
        ema_decay = raw.get("ema_decay")
        module_classes = raw.get("activation_checkpoint_module_classes", ())
        if not isinstance(module_classes, (list, tuple)):
            raise ValueError("activation_checkpoint_module_classes must be a list.")
        config = cls(
            strategy=strategy,
            backend=str(raw.get("backend", "auto")).casefold(),
            init_method=str(raw.get("init_method", "env://")),
            timeout_seconds=int(raw.get("timeout_seconds", 600)),
            precision=precision,
            gradient_accumulation_steps=int(raw.get("gradient_accumulation_steps", 1)),
            max_grad_norm=None if max_grad_norm is None else float(max_grad_norm),
            find_unused_parameters=bool(raw.get("find_unused_parameters", False)),
            static_graph=bool(raw.get("static_graph", False)),
            activation_checkpointing=bool(raw.get("activation_checkpointing", False)),
            activation_checkpoint_module_classes=tuple(str(item) for item in module_classes),
            activation_checkpoint_min_params=int(
                raw.get("activation_checkpoint_min_params", 0)
            ),
            fsdp_sharding_strategy=sharding,
            fsdp_min_num_params=int(raw.get("fsdp_min_num_params", 100_000_000)),
            fsdp_cpu_offload=bool(raw.get("fsdp_cpu_offload", False)),
            fsdp_sync_module_states=bool(raw.get("fsdp_sync_module_states", True)),
            fsdp_use_orig_params=bool(raw.get("fsdp_use_orig_params", True)),
            ema_decay=None if ema_decay is None else float(ema_decay),
            ema_device=str(raw.get("ema_device", "model")).casefold(),
            checkpoint_every_steps=int(raw.get("checkpoint_every_steps", 0)),
            checkpoint_keep_last=int(raw.get("checkpoint_keep_last", 3)),
            checkpoint_strict_topology=bool(raw.get("checkpoint_strict_topology", False)),
        )
        config.validate(world_size=world_size)
        return config

    def validate(self, *, world_size: int = 1) -> None:
        if self.strategy not in _STRATEGIES:
            raise ValueError(f"strategy must be one of {sorted(_STRATEGIES)}.")
        if self.precision not in _PRECISIONS:
            raise ValueError(f"precision must be one of {sorted(_PRECISIONS)}.")
        if self.fsdp_sharding_strategy not in _SHARDING_STRATEGIES:
            raise ValueError(
                "fsdp_sharding_strategy must be one of "
                f"{sorted(_SHARDING_STRATEGIES)}."
            )
        if self.strategy == "single" and world_size > 1:
            raise ValueError("strategy=single cannot be used with world_size > 1.")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive.")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive.")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive or null.")
        if self.activation_checkpointing and not (
            self.activation_checkpoint_module_classes
            or self.activation_checkpoint_min_params > 0
        ):
            raise ValueError(
                "activation checkpointing requires module class names or a positive "
                "activation_checkpoint_min_params threshold."
            )
        if self.fsdp_min_num_params < 1:
            raise ValueError("fsdp_min_num_params must be positive.")
        if self.ema_decay is not None and not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must be between zero and one.")
        if self.ema_device not in {"model", "cpu"}:
            raise ValueError("ema_device must be model or cpu.")
        if self.checkpoint_every_steps < 0:
            raise ValueError("checkpoint_every_steps cannot be negative.")
        if self.checkpoint_keep_last < 1:
            raise ValueError("checkpoint_keep_last must be positive.")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["activation_checkpoint_module_classes"] = list(
            self.activation_checkpoint_module_classes
        )
        return value
