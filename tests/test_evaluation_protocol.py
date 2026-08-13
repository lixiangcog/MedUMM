import json

import pytest

from medumm.core import EvaluationMode
from medumm.evaluation import EvaluationProtocol, MedicalVQACoreMetrics
from medumm.evaluation.medical_vqa import MedicalVQABenchmark, _sample_parameters
from medumm.core.runtime import RuntimeContext
from tests.conftest import PROJECT_ROOT


def test_protocol_rejects_invalid_uncertainty_and_groups():
    with pytest.raises(ValueError, match="confidence_level"):
        EvaluationProtocol(confidence_level=1.0)
    with pytest.raises(ValueError, match="Unsupported"):
        EvaluationProtocol(group_by=("patient_id",))


def test_metric_suite_reports_closed_accuracy_and_seeded_intervals():
    suite = MedicalVQACoreMetrics()
    rows = [
        {
            "exact_match": float(value),
            "token_f1": float(value),
            "abstained": False,
            "choices": {"A": "yes", "B": "no"},
            "modality": "radiology",
        }
        for value in (1, 0, 1, 0)
    ]
    protocol = {
        "group_by": ["modality"],
        "bootstrap_samples": 100,
        "confidence_level": 0.95,
        "seed": 7,
    }
    first = suite.summarize(rows, protocol)
    second = suite.summarize(rows, protocol)
    assert first == second
    assert first["overall"]["closed_accuracy"] == 50.0
    assert first["uncertainty"]["exact_match"]["bootstrap_samples"] == 100


def test_choice_candidates_are_added_per_sample_without_mutating_model_defaults():
    defaults = {"temperature": 0.0}
    result = _sample_parameters(
        defaults,
        {"A": "normal chest x-ray", "B": "pneumonia chest x-ray"},
        use_choice_candidates=True,
    )
    assert result == {
        "temperature": 0.0,
        "candidates": ["normal chest x-ray", "pneumonia chest x-ray"],
    }
    assert defaults == {"temperature": 0.0}


def test_audit_mode_needs_no_model_and_writes_quality_report(tmp_path):
    runtime = RuntimeContext.create(
        command="evaluation",
        config_path=PROJECT_ROOT / "pyproject.toml",
        output_directory=tmp_path,
    )
    result = MedicalVQABenchmark().run(
        {
            "benchmark": "medical_vqa",
            "data": {
                "path": "examples/medical/tiny_eval.jsonl",
                "image_root": "examples/medical/images",
                "deidentified": True,
            },
            "output_directory": str(tmp_path),
            "protocol": {"minimum_samples": 3},
            "mode": EvaluationMode.AUDIT.value,
        },
        config_path=PROJECT_ROOT / "pyproject.toml",
        runtime=runtime,
    )
    audit = json.loads((tmp_path / "dataset_audit.json").read_text())
    assert result.dataset_size == 3
    assert audit["sample_count"] == 3
    assert audit["manifest"]["sha256"]
    assert audit["status"] in {"passed", "warning"}


def test_audit_enforces_governance_gates(tmp_path):
    runtime = RuntimeContext.create(
        command="evaluation",
        config_path=PROJECT_ROOT / "pyproject.toml",
        output_directory=tmp_path,
    )
    with pytest.raises(ValueError, match="requires deidentified"):
        MedicalVQABenchmark().run(
            {
                "data": {
                    "path": "examples/medical/tiny_eval.jsonl",
                    "image_root": "examples/medical/images",
                },
                "output_directory": str(tmp_path),
                "protocol": {"require_deidentified": True},
                "mode": "audit",
            },
            config_path=PROJECT_ROOT / "pyproject.toml",
            runtime=runtime,
        )


def test_slurm_environment_selects_rank_local_stride(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_PROCID", "1")
    monkeypatch.setenv("SLURM_LOCALID", "1")
    monkeypatch.setenv("SLURM_NTASKS", "2")
    runtime = RuntimeContext.create(
        command="evaluation",
        config_path=PROJECT_ROOT / "pyproject.toml",
        output_directory=tmp_path,
    )
    result = MedicalVQABenchmark().run(
        {
            "data": {
                "path": "examples/medical/tiny_eval.jsonl",
                "image_root": "examples/medical/images",
            },
            "model": {
                "backbone": "medical_reference",
                "parameters": {"fixed_answer": "A"},
            },
            "output_directory": str(tmp_path),
            "mode": "full",
            "resume": False,
        },
        config_path=PROJECT_ROOT / "pyproject.toml",
        runtime=runtime,
    )
    assert result.dataset_size == 1
    assert (tmp_path / "predictions.rank-00001-of-00002.jsonl").is_file()
