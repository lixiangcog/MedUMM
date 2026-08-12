import argparse
import json
from importlib.util import module_from_spec, spec_from_file_location

import pytest

from tests.conftest import PROJECT_ROOT


def _module():
    path = PROJECT_ROOT / "scripts/verify_real_model_run.py"
    spec = spec_from_file_location("verify_real_model_run", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_real_model_verifier_accepts_complete_gpu_evidence(tmp_path):
    module = _module()
    inference = tmp_path / "inference.json"
    predictions = tmp_path / "predictions.jsonl"
    score = tmp_path / "score.json"
    assets = tmp_path / "assets.json"
    dataset = tmp_path / "dataset.json"
    output = tmp_path / "evidence.json"
    metadata = {
        "device": "cuda:0",
        "dtype": "float16",
        "model_family": "llava_mistral",
        "peak_gpu_memory_mb": 1000.0,
    }
    inference.write_text(json.dumps([{
        "request_id": "one",
        "model_name": "llava_med",
        "text": "yes",
        "duration_ms": 10.0,
        "metadata": metadata,
    }]))
    predictions.write_text(json.dumps({"id": "one", "prediction": "yes"}) + "\n")
    score.write_text(json.dumps({
        "metrics": {"overall": {"exact_match": 100.0}},
        "metadata": {"inference": {"mean_duration_ms": 10.0, "max_peak_gpu_memory_mb": 1000.0}},
    }))
    assets.write_text(json.dumps({"model_id": "microsoft/test", "model_revision": "abc"}))
    dataset.write_text(json.dumps({
        "dataset": "vqa-rad",
        "resolved_revision": "def",
        "split": "test",
        "sample_count": 1,
        "license": "CC0-1.0",
    }))
    evidence = module.verify(argparse.Namespace(
        inference=inference,
        predictions=predictions,
        score=score,
        asset_provenance=assets,
        dataset_provenance=dataset,
        output=output,
    ))
    assert evidence["status"] == "passed"
    assert evidence["runtime"]["python"]
    assert output.is_file()


def test_real_model_verifier_rejects_truncated_non_answer(tmp_path):
    module = _module()
    inference = tmp_path / "inference.json"
    predictions = tmp_path / "predictions.jsonl"
    score = tmp_path / "score.json"
    assets = tmp_path / "assets.json"
    dataset = tmp_path / "dataset.json"
    inference.write_text(json.dumps([{
        "request_id": "one",
        "model_name": "llava_med",
        "text": "The",
        "duration_ms": 10.0,
        "metadata": {
            "device": "cuda:0",
            "dtype": "float16",
            "model_family": "llava_mistral",
            "peak_gpu_memory_mb": 1000.0,
        },
    }]))
    predictions.write_text(json.dumps({"id": "one", "prediction": "yes"}) + "\n")
    score.write_text(json.dumps({
        "metrics": {},
        "metadata": {"inference": {"mean_duration_ms": 10.0, "max_peak_gpu_memory_mb": 1000.0}},
    }))
    assets.write_text(json.dumps({"model_id": "microsoft/test", "model_revision": "abc"}))
    dataset.write_text(json.dumps({
        "dataset": "vqa-rad",
        "resolved_revision": "def",
        "split": "test",
        "sample_count": 1,
        "license": "CC0-1.0",
    }))
    with pytest.raises(AssertionError, match="yes or no"):
        module.verify(argparse.Namespace(
            inference=inference,
            predictions=predictions,
            score=score,
            asset_provenance=assets,
            dataset_provenance=dataset,
            output=tmp_path / "evidence.json",
        ))
