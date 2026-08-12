import argparse
import json

import pytest

from scripts.verify_medical_tasks_v0_6 import verify


def _dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path):
    tasks = {
        "finding_assessment",
        "clinical_description",
        "anatomy_localization",
        "quantitative_assessment",
        "image_context",
        "diagnostic_reasoning",
    }
    predictions = tmp_path / "predictions.jsonl"
    rows = []
    for task in sorted(tasks):
        for index in range(4):
            rows.append(
                {
                    "id": f"{task}-{index}",
                    "prediction": "yes" if task == "finding_assessment" else "valid answer",
                    "model_metadata": {
                        "device": "cuda:0",
                        "dtype": "float16",
                        "medical_task": task,
                        "generated_tokens": 4,
                        "keyword_stopping": False,
                        "hostname": "gpu-test",
                        "scheduler": {
                            "slurm_job_id": "1",
                            "slurm_step_id": "2",
                        },
                    },
                }
            )
    predictions.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    protocol = {
        "name": "vqa_rad_medical_tasks",
        "metric_suite": "medical_task_core",
        "metric_suite_version": "1.0",
    }
    metrics = {
        "overall": {
            "task_success": 25.0,
            "strict_diagnostic_accuracy": 75.0,
        },
        "by_medical_task": {task: {"total": 4} for task in tasks},
        "uncertainty": {
            "task_success": {"method": "wilson"},
            "concept_recall": {"method": "bootstrap"},
        },
    }
    score = tmp_path / "score.json"
    _dump(
        score,
        {
            "dataset_size": 24,
            "metadata": {
                "dataset_fingerprint": "stable",
                "protocol": protocol,
                "inference": {
                    "max_peak_gpu_memory_mb": 10,
                    "mean_generated_tokens": 4,
                    "model": {"device": "cuda:0", "dtype": "float16"},
                },
            },
            "metrics": metrics,
        },
    )
    audit = tmp_path / "audit.json"
    _dump(
        audit,
        {
            "status": "warning",
            "errors": [],
            "dataset_fingerprint": "stable",
            "task_distribution": {task: 4 for task in tasks},
            "mapping": {"heuristic_count": 24},
            "reference_provenance": {"present": 24},
            "case_count": 12,
            "samples_without_case_id": 0,
        },
    )
    assets = tmp_path / "assets.json"
    _dump(assets, {"model_id": "model", "model_revision": "revision"})
    dataset = tmp_path / "dataset.json"
    _dump(
        dataset,
        {
            "dataset": "dataset",
            "resolved_revision": "revision",
            "split": "test",
            "sample_count": 24,
            "license": "test",
            "task_mapping": {
                "selected_distribution": {task: 4 for task in tasks}
            },
        },
    )
    return argparse.Namespace(
        predictions=predictions,
        score=score,
        audit=audit,
        asset_provenance=assets,
        dataset_provenance=dataset,
        output=tmp_path / "evidence.json",
    )


def test_v06_verifier_accepts_task_aware_cuda_evidence(tmp_path):
    arguments = _fixture(tmp_path)
    result = verify(arguments)
    assert result["release"] == "v0.6.0"
    assert result["status"] == "passed"
    assert arguments.output.is_file()


def test_v06_verifier_rejects_open_answer_truncation(tmp_path):
    arguments = _fixture(tmp_path)
    rows = [json.loads(line) for line in arguments.predictions.read_text().splitlines()]
    next(
        row
        for row in rows
        if row["model_metadata"]["medical_task"] == "clinical_description"
    )["model_metadata"]["generated_tokens"] = 1
    arguments.predictions.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(AssertionError, match="truncated"):
        verify(arguments)
