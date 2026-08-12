#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any


EXPECTED_TASKS = {
    "finding_assessment",
    "clinical_description",
    "anatomy_localization",
    "quantitative_assessment",
    "image_context",
    "diagnostic_reasoning",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def verify(arguments: argparse.Namespace) -> dict[str, Any]:
    predictions = _read_jsonl(arguments.predictions)
    score = _read_json(arguments.score)
    audit = _read_json(arguments.audit)
    dataset = _read_json(arguments.dataset_provenance)
    assets = _read_json(arguments.asset_provenance)
    expected = int(dataset["sample_count"])
    if expected != 24 or len(predictions) != expected or score.get("dataset_size") != expected:
        raise AssertionError("The v0.6 acceptance slice must contain 24 aligned predictions.")
    if audit.get("status") not in {"passed", "warning"} or audit.get("errors"):
        raise AssertionError("Medical task dataset audit did not pass.")
    if audit.get("dataset_fingerprint") != score.get("metadata", {}).get(
        "dataset_fingerprint"
    ):
        raise AssertionError("Dataset audit and score fingerprints differ.")
    if set(audit.get("task_distribution", {})) != EXPECTED_TASKS:
        raise AssertionError("The real acceptance slice does not cover six medical tasks.")
    if set(audit.get("task_distribution", {}).values()) != {4}:
        raise AssertionError("The real medical task slice is not balanced at four samples each.")
    if audit.get("mapping", {}).get("heuristic_count") != expected:
        raise AssertionError("Heuristic task mappings are not explicitly disclosed.")
    if audit.get("reference_provenance", {}).get("present") != expected:
        raise AssertionError("Reference provenance is incomplete.")
    if audit.get("case_count", 0) < 1 or audit.get("samples_without_case_id"):
        raise AssertionError("Medical case identity is incomplete.")

    protocol = score.get("metadata", {}).get("protocol", {})
    if protocol.get("name") != "vqa_rad_medical_tasks":
        raise AssertionError("Unexpected v0.6 medical task protocol.")
    if protocol.get("metric_suite") != "medical_task_core":
        raise AssertionError("Unexpected v0.6 metric suite.")
    if protocol.get("metric_suite_version") != "1.0":
        raise AssertionError("Medical task metric suite version is not pinned.")
    metrics = score.get("metrics", {})
    if set(metrics.get("by_medical_task", {})) != EXPECTED_TASKS:
        raise AssertionError("Per-task metrics are incomplete.")
    if metrics.get("overall", {}).get("task_success") is None:
        raise AssertionError("Medical task success metric is missing.")
    if metrics.get("overall", {}).get("strict_diagnostic_accuracy") is None:
        raise AssertionError("Strict diagnosis metric is missing.")
    uncertainty = metrics.get("uncertainty", {})
    if uncertainty.get("task_success", {}).get("method") != "wilson":
        raise AssertionError("Task success must use a Wilson confidence interval.")
    if uncertainty.get("concept_recall", {}).get("method") != "bootstrap":
        raise AssertionError("Concept recall must use a bootstrap confidence interval.")
    if any(not str(row.get("prediction", "")).strip() for row in predictions):
        raise AssertionError("At least one real-model prediction is empty.")
    if any(
        row.get("model_metadata", {}).get("keyword_stopping") is not False
        for row in predictions
    ):
        raise AssertionError("The acceptance run did not use upstream-compatible EOS stopping.")
    long_form_rows = [
        row
        for row in predictions
        if row.get("model_metadata", {}).get("medical_task")
        in {"clinical_description", "diagnostic_reasoning"}
    ]
    if len(long_form_rows) != 8 or any(
        int(row.get("model_metadata", {}).get("generated_tokens", 0)) <= 2
        for row in long_form_rows
    ):
        raise AssertionError("An open medical answer was truncated to one generated word.")
    inference = score.get("metadata", {}).get("inference", {})
    model = inference.get("model", {})
    if not str(model.get("device", "")).startswith("cuda"):
        raise AssertionError("The v0.6 acceptance run did not use CUDA.")
    if inference.get("max_peak_gpu_memory_mb") is None:
        raise AssertionError("Peak GPU memory evidence is missing.")
    if inference.get("mean_generated_tokens") is None:
        raise AssertionError("Generated-token evidence is missing.")
    prediction_schedulers = {
        (
            row.get("model_metadata", {}).get("scheduler", {}).get("slurm_job_id"),
            row.get("model_metadata", {}).get("scheduler", {}).get("slurm_step_id"),
        )
        for row in predictions
    }
    if len(prediction_schedulers) != 1 or None in next(iter(prediction_schedulers)):
        raise AssertionError("Prediction Slurm provenance is missing or mixed.")
    prediction_job_id, prediction_step_id = next(iter(prediction_schedulers))
    prediction_hosts = {
        row.get("model_metadata", {}).get("hostname") for row in predictions
    }
    if len(prediction_hosts) != 1 or None in prediction_hosts:
        raise AssertionError("Prediction hostname provenance is missing or mixed.")

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
        "release": "v0.6.0",
        "status": "passed",
        "scheduler": {
            "slurm_job_id": prediction_job_id,
            "slurm_step_id": prediction_step_id,
            "verification_job_id": os.environ.get("SLURM_JOB_ID"),
            "verification_step_id": os.environ.get("SLURM_STEP_ID"),
        },
        "hostname": next(iter(prediction_hosts)),
        "verification_hostname": platform.node(),
        "runtime": runtime,
        "model": {
            "id": assets["model_id"],
            "revision": assets["model_revision"],
            "device": model["device"],
            "dtype": model["dtype"],
        },
        "dataset": {
            "id": dataset["dataset"],
            "revision": dataset["resolved_revision"],
            "split": dataset["split"],
            "sample_count": expected,
            "license": dataset["license"],
            "task_distribution": dataset["task_mapping"]["selected_distribution"],
            "task_mapping": dataset["task_mapping"],
            "audit_status": audit["status"],
        },
        "evaluation": {
            "protocol": protocol,
            "prediction_count": len(predictions),
            "metrics": metrics,
            "inference": inference,
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the MedUMM v0.6 medical task run")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--score", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--asset-provenance", required=True, type=Path)
    parser.add_argument("--dataset-provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(arguments: list[str] | None = None) -> int:
    values = build_parser().parse_args(arguments)
    print(json.dumps(verify(values), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
