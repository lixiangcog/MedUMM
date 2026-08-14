from __future__ import annotations

from functools import partial
from typing import Any

from medumm.training.config import DistributedTrainingConfig
from medumm.training.distributed import DistributedSession


def _apply_activation_checkpointing(model: Any, config: DistributedTrainingConfig) -> None:
    if not config.activation_checkpointing:
        return
    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "The installed PyTorch version does not provide activation checkpointing."
        ) from error

    names = set(config.activation_checkpoint_module_classes)
    threshold = config.activation_checkpoint_min_params

    def check_fn(module: Any) -> bool:
        if names and module.__class__.__name__ in names:
            return True
        return threshold > 0 and sum(
            parameter.numel() for parameter in module.parameters(recurse=False)
        ) >= threshold

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        ),
        check_fn=check_fn,
    )


def wrap_model(
    model: Any,
    *,
    session: DistributedSession,
    config: DistributedTrainingConfig,
) -> Any:
    """Move and wrap a model using the selected single/DDP/FSDP strategy."""

    import torch

    _apply_activation_checkpointing(model, config)
    if config.strategy == "single":
        return model.to(session.device)
    if config.strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel

        model = model.to(session.device)
        device_ids = [session.local_rank] if session.device.type == "cuda" else None
        output_device = session.local_rank if session.device.type == "cuda" else None
        return DistributedDataParallel(
            model,
            device_ids=device_ids,
            output_device=output_device,
            find_unused_parameters=config.find_unused_parameters,
            static_graph=config.static_graph,
        )
    if config.strategy != "fsdp":
        raise ValueError(f"Unsupported distributed strategy: {config.strategy}")

    from torch.distributed.fsdp import (
        CPUOffload,
        FullyShardedDataParallel,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    sharding = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[config.fsdp_sharding_strategy]
    mixed_precision = None
    if config.precision in {"fp16", "bf16"}:
        dtype = torch.float16 if config.precision == "fp16" else torch.bfloat16
        mixed_precision = MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=dtype,
            buffer_dtype=dtype,
        )
    auto_wrap_policy = partial(
        size_based_auto_wrap_policy,
        min_num_params=config.fsdp_min_num_params,
    )
    return FullyShardedDataParallel(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding,
        cpu_offload=CPUOffload(offload_params=config.fsdp_cpu_offload),
        mixed_precision=mixed_precision,
        device_id=session.device if session.device.type == "cuda" else None,
        sync_module_states=config.fsdp_sync_module_states and session.world_size > 1,
        use_orig_params=config.fsdp_use_orig_params,
        limit_all_gathers=True,
    )


def unwrap_model(model: Any) -> Any:
    return getattr(model, "module", model)


def clip_grad_norm(model: Any, max_norm: float) -> float:
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel

    if isinstance(model, FullyShardedDataParallel):
        return float(model.clip_grad_norm_(max_norm).item())
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item())
