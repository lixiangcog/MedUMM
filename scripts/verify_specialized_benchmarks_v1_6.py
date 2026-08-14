#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any

from medumm.core.builtins import register_builtins
from medumm.core.registry import registry
from medumm.evaluation.benchmark_catalog import SPECIALIZED_BENCHMARKS
from medumm.evaluation.metrics import create_metric_suite
from medumm.resources import DATASET_RESOURCES


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"Expected a JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def verify(
    output_root: Path,
    evidence_path: Path,
    *,
    require_cuda: bool = False,
) -> dict[str, Any]:
    register_builtins()
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if not slurm_job_id:
        raise AssertionError("v1.6 acceptance must execute inside a Slurm allocation.")
    registered = set(registry.benchmarks.names())
    expected = {spec.name for spec in SPECIALIZED_BENCHMARKS}
    if not expected <= registered:
        raise AssertionError(f"Missing benchmark registrations: {sorted(expected - registered)}")

    rows = []
    total_predictions = 0
    for spec in SPECIALIZED_BENCHMARKS:
        directory = output_root / spec.name
        audit = _json(directory / "dataset_audit.json")
        score = _json(directory / "score.json")
        predictions = _jsonl(directory / "predictions.jsonl")
        results = _jsonl(directory / "results.jsonl")
        if audit.get("status") != "passed" or audit.get("errors"):
            raise AssertionError(f"{spec.name} dataset audit did not pass.")
        if score.get("benchmark") != spec.name:
            raise AssertionError(f"{spec.name} score has the wrong benchmark identity.")
        expected_size = int(score.get("dataset_size", 0))
        if expected_size < 1 or len(predictions) != expected_size or len(results) != expected_size:
            raise AssertionError(f"{spec.name} does not have aligned predictions and results.")
        report_metadata = dict(score.get("metadata", {}))
        benchmark_spec = dict(report_metadata.get("benchmark_spec", {}))
        protocol = dict(report_metadata.get("protocol", {}))
        if benchmark_spec.get("metric_suite") != spec.metric_suite:
            raise AssertionError(f"{spec.name} score did not preserve its fixed metric suite.")
        if protocol.get("metric_suite") != spec.metric_suite:
            raise AssertionError(f"{spec.name} protocol changed its fixed metric suite.")
        if create_metric_suite(spec.metric_suite).version != spec.version:
            raise AssertionError(f"{spec.name} benchmark and metric versions diverged.")
        if not score.get("metrics"):
            raise AssertionError(f"{spec.name} produced no aggregate metrics.")
        total_predictions += expected_size
        rows.append(
            {
                "benchmark": spec.name,
                "benchmark_version": spec.version,
                "metric_suite": spec.metric_suite,
                "dataset_size": expected_size,
                "audit_status": audit["status"],
                "metrics": score["metrics"],
                "output_directory": str(directory),
            }
        )

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        runtime = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": cuda_available,
            "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        }
    except ModuleNotFoundError:
        runtime = {
            "python": platform.python_version(),
            "torch": None,
            "cuda_version": None,
            "cuda_available": False,
            "gpu": None,
        }
    if require_cuda and not runtime["cuda_available"]:
        raise AssertionError("The Slurm verification process cannot see its allocated GPU.")

    evidence = {
        "schema_version": "1.0",
        "release": "v1.6.0",
        "status": "passed",
        "hostname": platform.node(),
        "scheduler": {"slurm_job_id": slurm_job_id},
        "runtime": runtime,
        "counts": {
            "registered_benchmarks": len(registered),
            "independent_benchmarks": len(registered - {"cross_task"}),
            "specialized_medical_benchmarks": len(rows),
            "dataset_resources": len(DATASET_RESOURCES.values()),
            "evaluated_fixture_samples": total_predictions,
        },
        "benchmarks": rows,
        "validation_scope": {
            "validated": (
                "All specialized benchmark contracts executed data audit, inference, dedicated "
                "scoring, and report generation in one Slurm allocation."
            ),
            "execution_model": "medical_reference deterministic software adapter",
            "hardware": (
                "CUDA allocation and visibility preflight"
                if require_cuda
                else "CPU Slurm allocation; the deterministic reference adapter does not require a GPU"
            ),
            "not_claimed": (
                "The synthetic fixture is not a real-dataset or real-model medical quality, "
                "fairness, robustness, or safety estimate."
            ),
        },
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify MedUMM v1.6 benchmark matrix")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    return parser


def main(arguments: list[str] | None = None) -> int:
    values = build_parser().parse_args(arguments)
    print(
        json.dumps(
            verify(
                values.output_root,
                values.output,
                require_cuda=values.require_cuda,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
