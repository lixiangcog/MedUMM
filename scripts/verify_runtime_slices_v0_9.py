#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any


MODEL_REVISIONS = {
    "lingshu_7b": "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9",
    "pubmedclip": "26c0c67f6da303ad2a38909130bd35744ea93517",
}
DATASET_REVISIONS = {
    "slake": "a9083ce6c34ac3ffb17671a605962924d8a8f9e9",
    "pneumoniamnist": "v2",
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"Expected a JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _validate_slice(
    *,
    model: str,
    dataset: str,
    predictions_path: Path,
    score_path: Path,
    audit_path: Path,
    provenance_path: Path,
    minimum_samples: int,
) -> dict[str, Any]:
    predictions = _jsonl(predictions_path)
    score = _json(score_path)
    audit = _json(audit_path)
    provenance = _json(provenance_path)
    expected = int(score.get("dataset_size", 0))
    if expected < minimum_samples or len(predictions) != expected:
        raise AssertionError(f"{model}/{dataset} did not produce the required predictions.")
    if audit.get("status") not in {"passed", "warning"} or audit.get("errors"):
        raise AssertionError(f"{dataset} audit failed.")
    if provenance.get("resolved_revision") != DATASET_REVISIONS[dataset]:
        raise AssertionError(f"{dataset} provenance is not pinned to the accepted release.")
    if any(row.get("model_name") != model for row in predictions):
        raise AssertionError(f"{model} prediction identity is inconsistent.")
    if any(not str(row.get("prediction", "")).strip() for row in predictions):
        raise AssertionError(f"{model} produced an empty prediction.")
    metadata = [dict(row.get("model_metadata", {})) for row in predictions]
    if any(value.get("model_revision") != MODEL_REVISIONS[model] for value in metadata):
        raise AssertionError(f"{model} revision evidence is missing or mixed.")
    if any(not str(value.get("device", "")).startswith("cuda") for value in metadata):
        raise AssertionError(f"{model} acceptance did not execute on CUDA.")
    job_ids = {value.get("scheduler", {}).get("slurm_job_id") for value in metadata}
    if model == "lingshu_7b" and (len(job_ids) != 1 or None in job_ids):
        raise AssertionError("Lingshu prediction Slurm provenance is missing or mixed.")
    inference = score.get("metadata", {}).get("inference", {})
    if inference.get("mean_duration_ms") is None:
        raise AssertionError(f"{model} latency evidence is missing.")
    if inference.get("max_peak_gpu_memory_mb") is None:
        raise AssertionError(f"{model} GPU memory evidence is missing.")
    return {
        "model": model,
        "model_revision": MODEL_REVISIONS[model],
        "dataset": dataset,
        "dataset_revision": DATASET_REVISIONS[dataset],
        "sample_count": expected,
        "metrics": score.get("metrics", {}),
        "inference": inference,
        "audit_status": audit["status"],
    }


def verify(arguments: argparse.Namespace) -> dict[str, Any]:
    lingshu = _validate_slice(
        model="lingshu_7b",
        dataset="slake",
        predictions_path=arguments.lingshu_predictions,
        score_path=arguments.lingshu_score,
        audit_path=arguments.lingshu_audit,
        provenance_path=arguments.slake_provenance,
        minimum_samples=4,
    )
    pubmedclip = _validate_slice(
        model="pubmedclip",
        dataset="pneumoniamnist",
        predictions_path=arguments.pubmedclip_predictions,
        score_path=arguments.pubmedclip_score,
        audit_path=arguments.pubmedclip_audit,
        provenance_path=arguments.pneumoniamnist_provenance,
        minimum_samples=16,
    )
    runtime: dict[str, Any] = {"python": platform.python_version()}
    try:
        import torch

        runtime.update(
            {
                "torch": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except ImportError:
        runtime["torch"] = None
    evidence = {
        "schema_version": "1.0",
        "release": "v0.9.0",
        "status": "passed",
        "scheduler": {"verification_job_id": os.environ.get("SLURM_JOB_ID")},
        "hostname": platform.node(),
        "runtime": runtime,
        "slices": [lingshu, pubmedclip],
        "validation_scope": (
            "Two pinned medical model/dataset slices through MedUMM evaluation; "
            "not a claim that every catalog entry is runtime validated."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify MedUMM v0.9 runtime slices")
    parser.add_argument("--lingshu-predictions", type=Path, required=True)
    parser.add_argument("--lingshu-score", type=Path, required=True)
    parser.add_argument("--lingshu-audit", type=Path, required=True)
    parser.add_argument("--slake-provenance", type=Path, required=True)
    parser.add_argument("--pubmedclip-predictions", type=Path, required=True)
    parser.add_argument("--pubmedclip-score", type=Path, required=True)
    parser.add_argument("--pubmedclip-audit", type=Path, required=True)
    parser.add_argument("--pneumoniamnist-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(arguments: list[str] | None = None) -> int:
    values = build_parser().parse_args(arguments)
    print(json.dumps(verify(values), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
