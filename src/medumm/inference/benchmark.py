from __future__ import annotations

import math
import platform
import statistics
import time
from pathlib import Path
from typing import Any

from medumm.core.config import execution_config
from medumm.core.io import ensure_directory, write_json
from medumm.core.runtime import RuntimeContext, environment_snapshot
from medumm.inference.pipeline import InferencePipeline


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _requests(block: dict[str, Any]) -> list[dict[str, Any]]:
    raw = block.get("requests", block.get("request"))
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list) or not raw or not all(isinstance(item, dict) for item in raw):
        raise ValueError("Inference benchmark requires one or more inference.requests.")
    return [dict(item) for item in raw]


def _iteration_requests(
    requests: list[dict[str, Any]], iteration: int, output_directory: Path
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for index, request in enumerate(requests):
        value = {**request}
        original = str(value.get("request_id", value.get("id", f"request-{index:04d}")))
        value["request_id"] = f"{original}-iteration-{iteration:04d}"
        if value.get("output_path"):
            raw = Path(str(value["output_path"]))
            value["output_path"] = str(
                output_directory / "artifacts" / f"iteration-{iteration:04d}" / raw.name
            )
        values.append(value)
    return values


def run_inference_benchmark(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    runtime: RuntimeContext | None = None,
) -> dict[str, Any]:
    inference = execution_config(config, "inference")
    benchmark = config.get("benchmark", {}) or {}
    if not isinstance(benchmark, dict):
        raise ValueError("benchmark must be a mapping.")
    backbone = str(inference.get("backbone", "")).strip().casefold()
    if not backbone:
        raise ValueError("Inference benchmark requires inference.backbone.")
    warmup_iterations = int(benchmark.get("warmup_iterations", 1))
    measured_iterations = int(benchmark.get("measured_iterations", 3))
    batch_size = int(benchmark.get("batch_size", inference.get("batch_size", 1)))
    if min(warmup_iterations, measured_iterations, batch_size) < 0 or measured_iterations < 1 or batch_size < 1:
        raise ValueError(
            "warmup_iterations must be non-negative; measured_iterations and batch_size must be positive."
        )
    context = runtime or RuntimeContext.create(
        command="benchmark_inference",
        config_path=config_path,
        output_directory=benchmark.get("output_directory"),
        runtime_config=config.get("runtime"),
    )
    output_directory = ensure_directory(
        context.output_directory
        if benchmark.get("output_directory") is None
        else (
            Path(str(benchmark["output_directory"]))
            if Path(str(benchmark["output_directory"])).is_absolute()
            else context.project_root / str(benchmark["output_directory"])
        )
    )
    base_requests = _requests(inference)
    durations_ms: list[float] = []
    engine_latencies_ms: list[float] = []
    ttft_ms: list[float] = []
    queue_ms: list[float] = []
    generated_tokens = 0
    iterations: list[dict[str, Any]] = []
    with InferencePipeline(
        backbone,
        dict(inference.get("config", {})),
        runtime=context,
    ) as pipeline:
        runtime_info = pipeline.runtime_info()
        for iteration in range(-warmup_iterations, measured_iterations):
            requests = _iteration_requests(base_requests, iteration, output_directory)
            started = time.perf_counter()
            results = pipeline.run_many(requests, batch_size=batch_size)
            wall_ms = (time.perf_counter() - started) * 1000
            if iteration < 0:
                continue
            iteration_tokens = sum(
                int(result.metadata.get("generated_tokens") or 0) for result in results
            )
            generated_tokens += iteration_tokens
            durations_ms.extend(
                float(result.metadata.get("engine_latency_ms") or result.duration_ms or wall_ms)
                for result in results
            )
            for result in results:
                if result.metadata.get("engine_latency_ms") is not None:
                    engine_latencies_ms.append(float(result.metadata["engine_latency_ms"]))
                if result.metadata.get("time_to_first_token_ms") is not None:
                    ttft_ms.append(float(result.metadata["time_to_first_token_ms"]))
                if result.metadata.get("queue_ms") is not None:
                    queue_ms.append(float(result.metadata["queue_ms"]))
            iterations.append(
                {
                    "iteration": iteration,
                    "requests": len(results),
                    "wall_time_ms": round(wall_ms, 3),
                    "generated_tokens": iteration_tokens,
                }
            )

    total_requests = len(base_requests) * measured_iterations
    total_seconds = sum(item["wall_time_ms"] for item in iterations) / 1000
    throughput = total_requests / total_seconds if total_seconds else None
    token_throughput = generated_tokens / total_seconds if total_seconds else None
    latency = {
        "mean_ms": round(statistics.fmean(durations_ms), 3) if durations_ms else None,
        "p50_ms": round(_percentile(durations_ms, 0.50), 3) if durations_ms else None,
        "p95_ms": round(_percentile(durations_ms, 0.95), 3) if durations_ms else None,
        "p99_ms": round(_percentile(durations_ms, 0.99), 3) if durations_ms else None,
    }
    report = {
        "schema_version": "1.0",
        "status": "completed",
        "backbone": backbone,
        "benchmark": {
            "warmup_iterations": warmup_iterations,
            "measured_iterations": measured_iterations,
            "requests_per_iteration": len(base_requests),
            "batch_size": batch_size,
            "total_requests": total_requests,
        },
        "throughput": {
            "requests_per_second": round(throughput, 6) if throughput is not None else None,
            "output_tokens_per_second": (
                round(token_throughput, 6) if token_throughput is not None else None
            ),
            "generated_tokens": generated_tokens,
            "measured_wall_seconds": round(total_seconds, 6),
        },
        "latency": latency,
        "engine": {
            "time_to_first_token_p50_ms": (
                round(_percentile(ttft_ms, 0.50), 3) if ttft_ms else None
            ),
            "queue_p50_ms": round(_percentile(queue_ms, 0.50), 3) if queue_ms else None,
            "latency_p50_ms": (
                round(_percentile(engine_latencies_ms, 0.50), 3)
                if engine_latencies_ms
                else None
            ),
        },
        "iterations": iterations,
        "runtime": runtime_info,
        "environment": environment_snapshot(context),
        "hostname": platform.node(),
        "clinical_use": False,
    }
    path = write_json(output_directory / "benchmark.json", report)
    report["output_path"] = str(path)
    return report
