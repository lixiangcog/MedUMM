#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verify(arguments: argparse.Namespace) -> dict[str, Any]:
    inference = _read_json(arguments.inference)
    predictions = _read_jsonl(arguments.predictions)
    score = _read_json(arguments.score)
    assets = _read_json(arguments.asset_provenance)
    dataset = _read_json(arguments.dataset_provenance)
    if not isinstance(inference, list) or len(inference) != 1:
        raise AssertionError("Expected exactly one direct inference result.")
    direct = inference[0]
    if direct.get("model_name") != "llava_med" or not str(direct.get("text", "")).strip():
        raise AssertionError("Direct inference did not contain a real LLaVA-Med answer.")
    if str(direct["text"]).strip().casefold().rstrip(".") not in {"yes", "no"}:
        raise AssertionError("Direct VQA smoke answer must resolve to yes or no.")
    metadata = direct.get("metadata", {})
    if not str(metadata.get("device", "")).startswith("cuda"):
        raise AssertionError(f"Expected CUDA inference, found {metadata.get('device')!r}.")
    if metadata.get("model_family") != "llava_mistral":
        raise AssertionError("Unexpected model family in inference metadata.")
    expected = int(dataset["sample_count"])
    if len(predictions) != expected:
        raise AssertionError(f"Expected {expected} predictions, found {len(predictions)}.")
    if any(not str(row.get("prediction", "")).strip() for row in predictions):
        raise AssertionError("At least one evaluation prediction is empty.")
    normalized_predictions = [
        str(row["prediction"]).strip().casefold().rstrip(".") for row in predictions
    ]
    if any(prediction not in {"yes", "no"} for prediction in normalized_predictions):
        raise AssertionError("Closed VQA smoke predictions must resolve to yes or no.")
    inference_summary = score.get("metadata", {}).get("inference", {})
    if inference_summary.get("mean_duration_ms") is None:
        raise AssertionError("Evaluation report is missing latency evidence.")
    if inference_summary.get("max_peak_gpu_memory_mb") is None:
        raise AssertionError("Evaluation report is missing GPU memory evidence.")
    runtime: dict[str, Any] = {
        "python": platform.python_version(),
    }
    try:
        import torch

        runtime.update({
            "torch": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_mb": round(
                torch.cuda.get_device_properties(0).total_memory / 1024**2,
                2,
            ) if torch.cuda.is_available() else None,
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
            "family": metadata["model_family"],
            "dtype": metadata["dtype"],
            "device": metadata["device"],
        },
        "dataset": {
            "id": dataset["dataset"],
            "revision": dataset["resolved_revision"],
            "split": dataset["split"],
            "sample_count": expected,
            "license": dataset["license"],
        },
        "direct_inference": {
            "request_id": direct["request_id"],
            "text": direct["text"],
            "duration_ms": direct["duration_ms"],
            "peak_gpu_memory_mb": metadata["peak_gpu_memory_mb"],
        },
        "evaluation": {
            "prediction_count": len(predictions),
            "metrics": score["metrics"],
            "inference": inference_summary,
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the MedUMM real-model GPU slice")
    parser.add_argument("--inference", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--score", required=True, type=Path)
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
