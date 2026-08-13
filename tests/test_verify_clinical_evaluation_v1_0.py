from __future__ import annotations

import argparse
import json

import pytest

from scripts.verify_clinical_evaluation_v1_0 import (
    LINGSHU_REVISION,
    PATHVQA_REVISION,
    verify,
)


def _arguments(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    score = tmp_path / "score.json"
    audit = tmp_path / "audit.json"
    provenance = tmp_path / "provenance.json"
    metadata = {
        "model_revision": LINGSHU_REVISION,
        "device": "cuda:0",
        "peak_gpu_memory_mb": 100.0,
    }
    predictions.write_text(
        "".join(
            json.dumps({"id": str(index), "prediction": "answer", "model_metadata": metadata}) + "\n"
            for index in range(8)
        )
    )
    score.write_text(json.dumps({
        "dataset_size": 8,
        "metrics": {"pathology": {
            "yes_no_accuracy": 50.0,
            "free_form_accuracy": 25.0,
            "overall_accuracy": 37.5,
            "macro_answer_type_accuracy": 37.5,
        }},
        "metadata": {"inference": {"mean_duration_ms": 10.0, "max_peak_gpu_memory_mb": 100.0}},
    }))
    audit.write_text(json.dumps({"status": "passed", "errors": []}))
    provenance.write_text(json.dumps({
        "resolved_revision": PATHVQA_REVISION,
        "answer_type_counts": {"closed": 4, "open": 4},
    }))
    return argparse.Namespace(
        predictions=predictions,
        score=score,
        audit=audit,
        provenance=provenance,
        output=tmp_path / "evidence.json",
    )


def test_v10_verifier_accepts_balanced_pathology_cuda_slice(tmp_path):
    arguments = _arguments(tmp_path)
    evidence = verify(arguments)
    assert evidence["status"] == "passed"
    assert evidence["slice"]["answer_type_counts"] == {"closed": 4, "open": 4}
    assert arguments.output.is_file()


def test_v10_verifier_rejects_unbalanced_answer_types(tmp_path):
    arguments = _arguments(tmp_path)
    arguments.provenance.write_text(json.dumps({
        "resolved_revision": PATHVQA_REVISION,
        "answer_type_counts": {"closed": 8, "open": 0},
    }))
    with pytest.raises(AssertionError, match="four closed and four open"):
        verify(arguments)
