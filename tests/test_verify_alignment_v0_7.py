from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.verify_alignment_v0_7 import (
    DATASET_ID,
    DATASET_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    verify,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_verify_alignment_accepts_audited_cuda_dpo_run(tmp_path: Path) -> None:
    preferences = tmp_path / "preferences.jsonl"
    preferences.write_text(
        "\n".join(json.dumps({"id": f"pair-{index}"}) for index in range(8)) + "\n",
        encoding="utf-8",
    )
    preference_sha = hashlib.sha256(preferences.read_bytes()).hexdigest()

    result = tmp_path / "result.json"
    _write_json(
        result,
        {
            "method": "medical_alignment",
            "status": "completed",
            "metrics": {
                "initial_loss": 0.69,
                "final_loss": 0.42,
                "initial_preference_accuracy": 0.0,
                "final_preference_accuracy": 0.75,
                "initial_reward_margin": 0.0,
                "final_reward_margin": 0.5,
            },
            "metadata": {
                "objective": "dpo",
                "global_steps": 2,
                "epochs": 1,
                "duration_seconds": 1.25,
                "device": "cuda",
                "dtype": "bfloat16",
                "quantization": "none",
                "peak_gpu_memory_mb": 256.0,
                "scheduler": {"slurm_job_id": "123", "slurm_step_id": "0"},
                "hostname": "gpu-test",
                "environment": {"cuda_available": True, "gpu_count": 1},
                "trainable_parameters": 8,
                "total_parameters": 80,
                "trainable_parameter_percent": 10.0,
            },
        },
    )
    checkpoint = tmp_path / "checkpoint.json"
    _write_json(
        checkpoint,
        {
            "format": "peft_adapter",
            "objective": "dpo",
            "base_model_revision": MODEL_REVISION,
            "dataset_fingerprint": "fingerprint",
        },
    )
    audit = tmp_path / "audit.json"
    _write_json(
        audit,
        {
            "status": "warning",
            "errors": [],
            "warnings": ["AI-judged labels."],
            "dataset_fingerprint": "fingerprint",
            "sample_count": 8,
            "preference_pair_count": 8,
            "preference_provenance_count": 8,
            "clinician_or_expert_count": 0,
        },
    )
    history = tmp_path / "history.jsonl"
    history.write_text(
        json.dumps({"step": 1, "gradient_norm": 1.0}) + "\n", encoding="utf-8"
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"test adapter")

    model_provenance = tmp_path / "model_provenance.json"
    _write_json(
        model_provenance,
        {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "license": "Apache-2.0",
        },
    )
    dataset_provenance = tmp_path / "dataset_provenance.json"
    _write_json(
        dataset_provenance,
        {
            "dataset": DATASET_ID,
            "resolved_revision": DATASET_REVISION,
            "license": "MIT",
            "deidentified": True,
            "sample_count": 8,
            "manifest_sha256": preference_sha,
            "selection": {"method": "test"},
            "preference_annotation": "AI judge; not a clinician annotation.",
        },
    )
    output = tmp_path / "evidence.json"

    evidence = verify(
        argparse.Namespace(
            result=result,
            checkpoint=checkpoint,
            audit=audit,
            history=history,
            adapter=adapter,
            model_provenance=model_provenance,
            dataset_provenance=dataset_provenance,
            preferences=preferences,
            output=output,
        )
    )

    assert evidence["status"] == "passed"
    assert evidence["training"]["global_steps"] == 2
    assert evidence["dataset"]["audit_status"] == "warning"
    assert output.exists()
