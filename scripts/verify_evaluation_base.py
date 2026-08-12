#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verify(arguments: argparse.Namespace) -> dict[str, Any]:
    predictions = _read_jsonl(arguments.predictions)
    score = _read_json(arguments.score)
    audit = _read_json(arguments.audit)
    dataset = _read_json(arguments.dataset_provenance)
    assets = _read_json(arguments.asset_provenance)
    expected = int(dataset["sample_count"])
    if len(predictions) != expected or score.get("dataset_size") != expected:
        raise AssertionError("Prediction, score, and provenance sample counts differ.")
    if audit.get("status") not in {"passed", "warning"} or audit.get("errors"):
        raise AssertionError("Medical dataset audit did not pass.")
    if audit.get("dataset_fingerprint") != score.get("metadata", {}).get("dataset_fingerprint"):
        raise AssertionError("Dataset audit and score fingerprints differ.")
    protocol = score.get("metadata", {}).get("protocol", {})
    if protocol.get("name") != "vqa_rad_closed":
        raise AssertionError("Unexpected medical evaluation protocol.")
    if protocol.get("metric_suite") != "medical_vqa_core":
        raise AssertionError("Unexpected medical metric suite.")
    if protocol.get("metric_suite_version") != "1.0":
        raise AssertionError("Metric suite version is not pinned.")
    metrics = score.get("metrics", {})
    if metrics.get("overall", {}).get("closed_accuracy") is None:
        raise AssertionError("Closed medical VQA accuracy is missing.")
    intervals = metrics.get("uncertainty", {})
    if set(intervals) != {"exact_match", "token_f1", "abstention_rate"}:
        raise AssertionError("Bootstrap uncertainty intervals are incomplete.")
    if any(not str(row.get("prediction", "")).strip() for row in predictions):
        raise AssertionError("At least one real-model prediction is empty.")
    inference = score.get("metadata", {}).get("inference", {})
    model = inference.get("model", {})
    if not str(model.get("device", "")).startswith("cuda"):
        raise AssertionError("Evaluation did not use CUDA.")
    if inference.get("max_peak_gpu_memory_mb") is None:
        raise AssertionError("Peak GPU memory evidence is missing.")

    runtime: dict[str, Any] = {"python": platform.python_version()}
    try:
        import torch

        runtime.update({
            "torch": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 2)
            if torch.cuda.is_available() else None,
        })
    except ImportError:
        runtime["torch"] = None
    evidence = {
        "schema_version": "1.0",
        "status": "passed",
        "scheduler": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        },
        "hostname": platform.node(),
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
    arguments.output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the MedUMM v0.4 evaluation base")
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
