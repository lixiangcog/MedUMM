from __future__ import annotations

import argparse
import json

import pytest

from scripts.verify_scale_catalog_v0_8 import (
    DATASET_REVISION,
    MODEL_NAME,
    MODEL_REVISION,
    verify,
)
from medumm.resources import DATASET_RESOURCES, MODEL_RESOURCES


def _fixture(tmp_path):
    catalog = tmp_path / "catalog.json"
    predictions = tmp_path / "predictions.jsonl"
    results = tmp_path / "results.jsonl"
    score = tmp_path / "score.json"
    audit = tmp_path / "audit.json"
    output = tmp_path / "evidence.json"
    catalog.write_text(
        json.dumps(
            {
                "valid": True,
                "registered_models": len(MODEL_RESOURCES.values()),
                "registered_datasets": len(DATASET_RESOURCES.values()),
            }
        )
    )
    metadata = {
        "resource": MODEL_NAME,
        "bridge_model": "llava_med",
        "model_revision": MODEL_REVISION,
        "device": "cuda:0",
        "peak_gpu_memory_mb": 100.0,
        "scheduler": {"slurm_job_id": "123", "slurm_step_id": "0"},
    }
    predictions.write_text(
        "".join(
            json.dumps(
                {
                    "id": f"case-{index}",
                    "model_name": MODEL_NAME,
                    "prediction": "yes",
                    "model_metadata": metadata,
                }
            )
            + "\n"
            for index in range(4)
        )
    )
    results.write_text(
        "".join(
            json.dumps(
                {
                    "id": f"case-{index}",
                    "sample_metadata": {
                        "resource": "vqa_rad",
                        "source_revision": DATASET_REVISION,
                    },
                }
            )
            + "\n"
            for index in range(4)
        )
    )
    score.write_text(
        json.dumps(
            {
                "dataset_size": 4,
                "metrics": {"overall": {"exact_match": 50.0}},
                "metadata": {
                    "dataset": "vqa_rad",
                    "model": MODEL_NAME,
                    "inference": {
                        "max_peak_gpu_memory_mb": 100.0,
                        "mean_duration_ms": 10.0,
                    },
                },
            }
        )
    )
    audit.write_text(json.dumps({"status": "passed", "errors": []}))
    return argparse.Namespace(
        catalog_validation=catalog,
        predictions=predictions,
        results=results,
        score=score,
        audit=audit,
        output=output,
    )


def test_v08_verifier_accepts_real_catalog_alias_evidence(tmp_path):
    arguments = _fixture(tmp_path)
    evidence = verify(arguments)
    assert evidence["status"] == "passed"
    assert evidence["catalog"]["model_count"] >= 20
    assert evidence["catalog"]["dataset_count"] >= 20
    assert arguments.output.is_file()


def test_v08_verifier_rejects_base_adapter_name(tmp_path):
    arguments = _fixture(tmp_path)
    rows = [json.loads(line) for line in arguments.predictions.read_text().splitlines()]
    rows[0]["model_name"] = "llava_med"
    arguments.predictions.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(AssertionError, match="catalog model"):
        verify(arguments)


def test_v08_verifier_accepts_batch_job_without_step_id(tmp_path):
    arguments = _fixture(tmp_path)
    rows = [json.loads(line) for line in arguments.predictions.read_text().splitlines()]
    for row in rows:
        row["model_metadata"]["scheduler"]["slurm_step_id"] = None
    arguments.predictions.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    evidence = verify(arguments)
    assert evidence["scheduler"]["slurm_job_id"] == "123"
    assert evidence["scheduler"]["slurm_step_id"] is None
