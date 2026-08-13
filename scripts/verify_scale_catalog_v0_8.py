#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any

from medumm.resources import DATASET_RESOURCES, MODEL_RESOURCES


MODEL_NAME = "llava_med_v1_5_7b"
DATASET_NAME = "vqa_rad"
MODEL_REVISION = "91bb16c122001ddc9cf1fd36ce1dae09448943a2"
DATASET_REVISION = "osf:5b213a9886d8510012c26c09:v1"


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


def verify(arguments: argparse.Namespace) -> dict[str, Any]:
    catalog = _json(arguments.catalog_validation)
    predictions = _jsonl(arguments.predictions)
    results = _jsonl(arguments.results)
    score = _json(arguments.score)
    audit = _json(arguments.audit)
    model_spec = MODEL_RESOURCES.get(MODEL_NAME)
    dataset_spec = DATASET_RESOURCES.get(DATASET_NAME)

    if catalog.get("valid") is not True:
        raise AssertionError("Resource catalog registration validation failed.")
    if catalog.get("registered_models") != len(MODEL_RESOURCES.values()):
        raise AssertionError("Not every model resource is registered.")
    if catalog.get("registered_datasets") != len(DATASET_RESOURCES.values()):
        raise AssertionError("Not every dataset resource is registered.")
    if len(MODEL_RESOURCES.values()) < 20 or len(DATASET_RESOURCES.values()) < 20:
        raise AssertionError("The v0.8 scale target requires at least 20 resources of each kind.")
    if model_spec.status.value != "runtime_validated":
        raise AssertionError("Acceptance model is not declared runtime_validated.")
    if dataset_spec.status.value != "runtime_validated":
        raise AssertionError("Acceptance dataset is not declared runtime_validated.")

    expected = int(score.get("dataset_size", 0))
    if expected < 4 or len(predictions) != expected or len(results) != expected:
        raise AssertionError("Real catalog acceptance requires at least four aligned samples.")
    if audit.get("status") not in {"passed", "warning"} or audit.get("errors"):
        raise AssertionError("Catalog dataset audit failed.")
    if any(row.get("model_name") != MODEL_NAME for row in predictions):
        raise AssertionError("Predictions did not use the catalog model registration.")
    if any(not str(row.get("prediction", "")).strip() for row in predictions):
        raise AssertionError("A real-model prediction is empty.")
    model_metadata = [dict(row.get("model_metadata", {})) for row in predictions]
    if any(value.get("resource") != MODEL_NAME for value in model_metadata):
        raise AssertionError("Model resource provenance is missing.")
    if any(value.get("bridge_model") != "llava_med" for value in model_metadata):
        raise AssertionError("Official-runtime bridge provenance is missing.")
    if any(value.get("model_revision") != MODEL_REVISION for value in model_metadata):
        raise AssertionError("Model revision is not pinned to the accepted release.")
    if any(not str(value.get("device", "")).startswith("cuda") for value in model_metadata):
        raise AssertionError("Acceptance inference did not use CUDA.")
    scheduler_pairs = {
        (
            value.get("scheduler", {}).get("slurm_job_id"),
            value.get("scheduler", {}).get("slurm_step_id"),
        )
        for value in model_metadata
    }
    if len(scheduler_pairs) != 1:
        raise AssertionError("Prediction Slurm provenance is mixed.")
    job_id, step_id = next(iter(scheduler_pairs))
    if not job_id:
        raise AssertionError("Prediction Slurm job provenance is missing.")

    for row in results:
        metadata = row.get("sample_metadata", {})
        if metadata.get("resource") != DATASET_NAME:
            raise AssertionError("Dataset resource provenance is missing from scored samples.")
        if metadata.get("source_revision") != DATASET_REVISION:
            raise AssertionError("Dataset revision is not pinned to the accepted release.")
    report_metadata = score.get("metadata", {})
    if report_metadata.get("dataset") != DATASET_NAME:
        raise AssertionError("Score report did not use the catalog dataset registration.")
    if report_metadata.get("model") != MODEL_NAME:
        raise AssertionError("Score report did not use the catalog model registration.")
    inference = report_metadata.get("inference", {})
    if inference.get("max_peak_gpu_memory_mb") is None:
        raise AssertionError("GPU memory evidence is missing.")
    if inference.get("mean_duration_ms") is None:
        raise AssertionError("Latency evidence is missing.")

    runtime: dict[str, Any] = {"python": platform.python_version()}
    try:
        import torch

        runtime.update(
            {
                "torch": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "gpu_memory_mb": round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**2, 2
                )
                if torch.cuda.is_available()
                else None,
            }
        )
    except ImportError:
        runtime["torch"] = None
    evidence = {
        "schema_version": "1.0",
        "release": "v0.8.0",
        "status": "passed",
        "validation_scope": {
            "catalog": "all resource schemas and individual plugin registrations",
            "runtime": "one pinned model plus one pinned dataset through catalog aliases",
            "not_claimed": "runtime execution of all cataloged heavyweight resources",
        },
        "scheduler": {
            "slurm_job_id": job_id,
            "slurm_step_id": step_id,
            "verification_job_id": os.environ.get("SLURM_JOB_ID"),
            "verification_step_id": os.environ.get("SLURM_STEP_ID"),
        },
        "hostname": platform.node(),
        "runtime": runtime,
        "catalog": {
            "version": MODEL_RESOURCES.version,
            "model_count": len(MODEL_RESOURCES.values()),
            "dataset_count": len(DATASET_RESOURCES.values()),
            "registered_model_count": catalog["registered_models"],
            "registered_dataset_count": catalog["registered_datasets"],
        },
        "acceptance_model": {
            "name": model_spec.name,
            "id": model_spec.artifact_id,
            "revision": MODEL_REVISION,
            "status": model_spec.status.value,
            "runtime_family": model_spec.runtime_family.value,
        },
        "acceptance_dataset": {
            "name": dataset_spec.name,
            "id": dataset_spec.artifact_id,
            "revision": DATASET_REVISION,
            "status": dataset_spec.status.value,
            "license": dataset_spec.license,
            "sample_count": expected,
            "audit_status": audit["status"],
        },
        "evaluation": {
            "prediction_count": len(predictions),
            "metrics": score.get("metrics", {}),
            "inference": inference,
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the MedUMM v0.8 scale catalog")
    parser.add_argument("--catalog-validation", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--score", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(arguments: list[str] | None = None) -> int:
    values = build_parser().parse_args(arguments)
    print(json.dumps(verify(values), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
