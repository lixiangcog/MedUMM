from __future__ import annotations

from pathlib import Path
from typing import Any

from medumm.core.builtins import register_builtins
from medumm.core.config import execution_config
from medumm.core.registry import registry
from medumm.core.results import EvaluationResult, InferenceResult, TrainingResult
from medumm.core.runtime import RuntimeContext
from medumm.inference import InferencePipeline
from medumm.post_training import PostTrainingRunner


def infer(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    runtime: RuntimeContext | None = None,
) -> list[InferenceResult]:
    """Run a canonical inference config through the public Python API."""

    block = execution_config(config, "inference")
    backbone = str(block.get("backbone", "")).strip().lower()
    if not backbone:
        raise ValueError("Inference config requires a backbone.")
    raw_requests = block.get("requests", block.get("request"))
    if raw_requests is None:
        raw_requests = [{
            key: block[key]
            for key in (
                "task",
                "medical_task",
                "prompt",
                "images",
                "videos",
                "parameters",
                "metadata",
                "output_path",
            )
            if key in block
        }]
    elif isinstance(raw_requests, dict):
        raw_requests = [raw_requests]
    if not isinstance(raw_requests, list) or not all(
        isinstance(request, dict) for request in raw_requests
    ):
        raise ValueError("Inference requests must be a list of mappings.")
    context = runtime or RuntimeContext.create(
        command="inference",
        config_path=config_path,
        output_directory=block.get("run_directory"),
        runtime_config=config.get("runtime"),
    )
    with InferencePipeline(
        backbone,
        dict(block.get("config", {})),
        runtime=context,
    ) as pipeline:
        return pipeline.run_many(
            raw_requests,
            batch_size=int(block.get("batch_size", 1)),
        )


def evaluate(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    runtime: RuntimeContext | None = None,
) -> EvaluationResult:
    """Run any registered benchmark through the public Python API."""

    register_builtins()
    block = execution_config(config, "evaluation")
    benchmark_name = str(block.get("benchmark", "medical_vqa")).strip().lower()
    context = runtime or RuntimeContext.create(
        command="evaluation",
        config_path=config_path,
        output_directory=block.get("output_directory"),
        runtime_config=config.get("runtime"),
    )
    benchmark = registry.benchmarks.create(benchmark_name)
    return benchmark.run(
        block,
        config_path=config_path or context.project_root,
        runtime=context,
    )


def post_train(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    runtime: RuntimeContext | None = None,
) -> TrainingResult:
    """Run any registered post-training method through the public Python API."""

    block = execution_config(config, "post_training")
    context = runtime or RuntimeContext.create(
        command="post_training",
        config_path=config_path,
        output_directory=block.get("output_directory"),
        runtime_config=config.get("runtime"),
    )
    return PostTrainingRunner(context).run(block, config_path=config_path)


def catalog() -> dict[str, list[dict[str, Any]]]:
    """Return all built-in components without importing heavy model libraries."""

    register_builtins()
    return registry.catalog()
