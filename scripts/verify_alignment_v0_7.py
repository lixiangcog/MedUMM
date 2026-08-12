#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


MODEL_ID = "EleutherAI/pythia-14m"
MODEL_REVISION = "cf967c0a9a04383db6f7b1108d86b2962634b4ac"
DATASET_ID = "TsinghuaC3I/UltraMedical-Preference"
DATASET_REVISION = "761eb7935310ba662a96d93c5af342e5269d5759"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(arguments: argparse.Namespace) -> dict[str, Any]:
    result = _json(arguments.result)
    checkpoint = _json(arguments.checkpoint)
    audit = _json(arguments.audit)
    model_assets = _json(arguments.model_provenance)
    data = _json(arguments.dataset_provenance)
    history = [
        json.loads(line)
        for line in arguments.history.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if result.get("method") != "medical_alignment" or result.get("status") != "completed":
        raise AssertionError("The v0.7 alignment run did not complete.")
    if checkpoint.get("format") != "peft_adapter" or checkpoint.get("objective") != "dpo":
        raise AssertionError("The v0.7 checkpoint is not a DPO PEFT adapter.")
    if checkpoint.get("base_model_revision") != MODEL_REVISION:
        raise AssertionError("The v0.7 base-model revision is not pinned.")
    if checkpoint.get("dataset_fingerprint") != audit.get("dataset_fingerprint"):
        raise AssertionError("Checkpoint and dataset-audit fingerprints differ.")
    if audit.get("status") not in {"passed", "warning"} or audit.get("errors"):
        raise AssertionError("The alignment data audit failed.")
    if audit.get("sample_count") != 8 or audit.get("preference_pair_count") != 8:
        raise AssertionError("The v0.7 acceptance run requires eight aligned pairs.")
    if audit.get("preference_provenance_count") != 8:
        raise AssertionError("Preference provenance is incomplete.")
    if audit.get("clinician_or_expert_count") != 0:
        raise AssertionError("AI-judged preferences were mislabeled as expert preferences.")
    if data.get("dataset") != DATASET_ID or data.get("resolved_revision") != DATASET_REVISION:
        raise AssertionError("Unexpected v0.7 preference dataset identity.")
    if data.get("license") != "MIT" or data.get("deidentified") is not True:
        raise AssertionError("Dataset license/de-identification evidence is incomplete.")
    if data.get("sample_count") != 8 or data.get("manifest_sha256") != _sha256(
        arguments.preferences
    ):
        raise AssertionError("Prepared preference manifest is not the pinned eight-row slice.")
    if model_assets.get("model_id") != MODEL_ID:
        raise AssertionError("Unexpected v0.7 base model.")
    if model_assets.get("model_revision") != MODEL_REVISION:
        raise AssertionError("Model asset revision is not pinned.")
    if model_assets.get("license") != "Apache-2.0":
        raise AssertionError("Base model license evidence is incomplete.")

    metadata = result.get("metadata", {})
    metrics = result.get("metrics", {})
    if metadata.get("objective") != "dpo" or metadata.get("global_steps", 0) < 1:
        raise AssertionError("No DPO optimization step was recorded.")
    if not str(metadata.get("device", "")).startswith("cuda"):
        raise AssertionError("The accepted alignment run did not use CUDA.")
    peak_memory = metadata.get("peak_gpu_memory_mb")
    if peak_memory is None or not math.isfinite(float(peak_memory)) or peak_memory <= 0:
        raise AssertionError("Peak GPU memory evidence is missing.")
    if not metadata.get("scheduler", {}).get("slurm_job_id"):
        raise AssertionError("Slurm job provenance is missing.")
    trainable = int(metadata.get("trainable_parameters", 0))
    total = int(metadata.get("total_parameters", 0))
    if not 0 < trainable < total:
        raise AssertionError("PEFT trainable-parameter evidence is invalid.")
    for key in (
        "initial_loss",
        "final_loss",
        "initial_preference_accuracy",
        "final_preference_accuracy",
        "initial_reward_margin",
        "final_reward_margin",
    ):
        if metrics.get(key) is None or not math.isfinite(float(metrics[key])):
            raise AssertionError(f"Training metric is missing: {key}")
    if not history or not all(
        math.isfinite(float(row.get("gradient_norm", 0)))
        and float(row.get("gradient_norm", 0)) > 0
        for row in history
    ):
        raise AssertionError("Training history does not prove non-zero gradients.")
    adapter_files = [
        path for path in arguments.adapter.rglob("*") if path.is_file()
    ]
    if not adapter_files or not any(path.suffix == ".safetensors" for path in adapter_files):
        raise AssertionError("Saved PEFT adapter weights are missing.")

    evidence = {
        "schema_version": "1.0",
        "release": "v0.7.0",
        "status": "passed",
        "scheduler": metadata["scheduler"],
        "hostname": metadata.get("hostname"),
        "runtime": metadata.get("environment"),
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": model_assets["license"],
            "total_parameters": total,
            "trainable_parameters": trainable,
            "trainable_parameter_percent": metadata.get("trainable_parameter_percent"),
        },
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "license": data["license"],
            "sample_count": data["sample_count"],
            "selection": data["selection"],
            "preference_annotation": data["preference_annotation"],
            "audit_status": audit["status"],
            "audit_warnings": audit["warnings"],
        },
        "training": {
            "objective": "dpo",
            "global_steps": metadata["global_steps"],
            "epochs": metadata["epochs"],
            "duration_seconds": metadata["duration_seconds"],
            "device": metadata["device"],
            "dtype": metadata["dtype"],
            "quantization": metadata["quantization"],
            "peak_gpu_memory_mb": metadata["peak_gpu_memory_mb"],
            "metrics": metrics,
            "history_sha256": _sha256(arguments.history),
            "adapter_files": [
                {
                    "path": str(path.relative_to(arguments.adapter)),
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in sorted(adapter_files)
            ],
        },
        "notes": [
            "This verifies software optimization and checkpoint reloadability, not clinical improvement.",
            "The eight upstream preferences are AI-judged and are not clinician annotations.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify MedUMM v0.7 alignment run")
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--model-provenance", required=True, type=Path)
    parser.add_argument("--dataset-provenance", required=True, type=Path)
    parser.add_argument("--preferences", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


if __name__ == "__main__":
    values = build_parser().parse_args()
    print(json.dumps(verify(values), indent=2, ensure_ascii=False))
