from __future__ import annotations

import argparse
import json

import pytest

from scripts.verify_runtime_slices_v0_9 import (
    DATASET_REVISIONS,
    MODEL_REVISIONS,
    verify,
)


def _slice(tmp_path, model, dataset, count):
    predictions = tmp_path / f"{model}-predictions.jsonl"
    score = tmp_path / f"{model}-score.json"
    audit = tmp_path / f"{dataset}-audit.json"
    provenance = tmp_path / f"{dataset}-provenance.json"
    metadata = {
        "model_revision": MODEL_REVISIONS[model],
        "device": "cuda:0",
        "peak_gpu_memory_mb": 100.0,
        "scheduler": {"slurm_job_id": "123", "slurm_step_id": None},
    }
    predictions.write_text(
        "".join(
            json.dumps(
                {
                    "id": f"{dataset}-{index}",
                    "model_name": model,
                    "prediction": "answer",
                    "model_metadata": metadata,
                }
            )
            + "\n"
            for index in range(count)
        )
    )
    score.write_text(
        json.dumps(
            {
                "dataset_size": count,
                "metrics": {"overall": {"exact_match": 50.0}},
                "metadata": {
                    "inference": {
                        "mean_duration_ms": 10.0,
                        "max_peak_gpu_memory_mb": 100.0,
                    }
                },
            }
        )
    )
    audit.write_text(json.dumps({"status": "passed", "errors": []}))
    provenance.write_text(
        json.dumps({"resolved_revision": DATASET_REVISIONS[dataset]})
    )
    return predictions, score, audit, provenance


def _arguments(tmp_path):
    lingshu = _slice(tmp_path, "lingshu_7b", "slake", 4)
    pubmedclip = _slice(tmp_path, "pubmedclip", "pneumoniamnist", 16)
    return argparse.Namespace(
        lingshu_predictions=lingshu[0],
        lingshu_score=lingshu[1],
        lingshu_audit=lingshu[2],
        slake_provenance=lingshu[3],
        pubmedclip_predictions=pubmedclip[0],
        pubmedclip_score=pubmedclip[1],
        pubmedclip_audit=pubmedclip[2],
        pneumoniamnist_provenance=pubmedclip[3],
        output=tmp_path / "evidence.json",
    )


def test_v09_verifier_accepts_two_pinned_cuda_slices(tmp_path):
    arguments = _arguments(tmp_path)
    evidence = verify(arguments)
    assert evidence["status"] == "passed"
    assert {item["model"] for item in evidence["slices"]} == {
        "lingshu_7b",
        "pubmedclip",
    }
    assert arguments.output.is_file()


def test_v09_verifier_rejects_mixed_model_revision(tmp_path):
    arguments = _arguments(tmp_path)
    rows = [json.loads(line) for line in arguments.lingshu_predictions.read_text().splitlines()]
    rows[0]["model_metadata"]["model_revision"] = "wrong"
    arguments.lingshu_predictions.write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    with pytest.raises(AssertionError, match="revision"):
        verify(arguments)
