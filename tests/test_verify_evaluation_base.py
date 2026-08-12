import argparse
import json

import pytest

from scripts.verify_evaluation_base import verify


def _dump(path, value):
    path.write_text(json.dumps(value))


def test_verifier_accepts_complete_evaluation_evidence(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"id": "one", "prediction": "yes"}) + "\n")
    protocol = {
        "name": "vqa_rad_closed",
        "metric_suite": "medical_vqa_core",
        "metric_suite_version": "1.0",
    }
    score = tmp_path / "score.json"
    _dump(score, {
        "dataset_size": 1,
        "metadata": {
            "dataset_fingerprint": "stable",
            "protocol": protocol,
            "inference": {
                "max_peak_gpu_memory_mb": 10,
                "model": {"device": "cuda:0", "dtype": "float16"},
            },
        },
        "metrics": {
            "overall": {"closed_accuracy": 100},
            "uncertainty": {
                name: {"lower": 100, "upper": 100}
                for name in ("exact_match", "token_f1", "abstention_rate")
            },
        },
    })
    audit = tmp_path / "audit.json"
    _dump(audit, {"status": "passed", "errors": [], "dataset_fingerprint": "stable"})
    assets = tmp_path / "assets.json"
    _dump(assets, {"model_id": "model", "model_revision": "revision"})
    dataset = tmp_path / "dataset.json"
    _dump(dataset, {
        "dataset": "dataset", "resolved_revision": "revision", "split": "test",
        "sample_count": 1, "license": "test",
    })
    output = tmp_path / "evidence.json"
    result = verify(argparse.Namespace(
        predictions=predictions, score=score, audit=audit,
        asset_provenance=assets, dataset_provenance=dataset, output=output,
    ))
    assert result["status"] == "passed"
    assert output.is_file()


def test_verifier_rejects_missing_uncertainty(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"id": "one", "prediction": "yes"}) + "\n")
    for name, value in {
        "score.json": {
            "dataset_size": 1,
            "metadata": {
                "dataset_fingerprint": "stable",
                "protocol": {"name": "vqa_rad_closed", "metric_suite": "medical_vqa_core", "metric_suite_version": "1.0"},
                "inference": {"max_peak_gpu_memory_mb": 10, "model": {"device": "cuda:0", "dtype": "float16"}},
            },
            "metrics": {"overall": {"closed_accuracy": 100}},
        },
        "audit.json": {"status": "passed", "errors": [], "dataset_fingerprint": "stable"},
        "assets.json": {"model_id": "model", "model_revision": "revision"},
        "dataset.json": {"dataset": "dataset", "resolved_revision": "revision", "split": "test", "sample_count": 1, "license": "test"},
    }.items():
        _dump(tmp_path / name, value)
    with pytest.raises(AssertionError, match="intervals"):
        verify(argparse.Namespace(
            predictions=predictions, score=tmp_path / "score.json", audit=tmp_path / "audit.json",
            asset_provenance=tmp_path / "assets.json", dataset_provenance=tmp_path / "dataset.json",
            output=tmp_path / "evidence.json",
        ))
