import json

import pytest

from medumm.core.runtime import RuntimeContext
from medumm.evaluation.medical_task_protocol import (
    MedicalTaskProtocol,
    audit_medical_task_dataset,
)
from medumm.evaluation.medical_tasks import MedicalTasksBenchmark
from medumm.medical.dataset import MedicalTasksDatasetAdapter
from medumm.medical.task_metrics import (
    evaluate_medical_task,
    summarize_medical_tasks,
)
from medumm.medical.tasks import MedicalTaskType, load_medical_tasks
from medumm.inference import InferenceRequest
from tests.conftest import PROJECT_ROOT


def _data_config():
    return {
        "path": "examples/medical/tiny_tasks.jsonl",
        "image_root": "examples/medical/images",
        "provenance": "examples/medical/provenance.json",
        "deidentified": True,
    }


def test_medical_task_schema_covers_perception_reasoning_and_generation():
    samples = load_medical_tasks(_data_config(), project_root=PROJECT_ROOT)
    assert len(samples) == len(MedicalTaskType) == 8
    assert {sample.task_family for sample in samples} == {
        "perception",
        "reasoning",
        "generation",
    }
    diagnosis = next(
        sample for sample in samples if sample.task is MedicalTaskType.DIAGNOSTIC_REASONING
    )
    assert diagnosis.concepts == ["synthetic bright-focus pattern"]
    assert diagnosis.evidence == ["bright central focus"]
    assert diagnosis.reference_provenance["kind"] == "synthetic_annotation"


def test_inference_request_keeps_execution_and_medical_semantics_separate():
    request = InferenceRequest(
        task="understanding",
        medical_task="report_generation",
        prompt="Draft a report from the image.",
    )
    assert request.task.value == "understanding"
    assert request.medical_task is MedicalTaskType.REPORT_GENERATION
    assert request.to_dict()["medical_task"] == "report_generation"
    with pytest.raises(ValueError, match="text-output understanding"):
        InferenceRequest(
            task="generation",
            medical_task="report_generation",
            prompt="This would mean image generation, not report text.",
        )


def test_medical_task_adapter_fingerprint_changes_with_selection():
    adapter = MedicalTasksDatasetAdapter()
    full = adapter.fingerprint(_data_config(), PROJECT_ROOT)
    limited = adapter.fingerprint({**_data_config(), "max_samples": 2}, PROJECT_ROOT)
    assert full != limited


def test_medical_task_protocol_rejects_unknown_task_and_group():
    with pytest.raises(ValueError, match="Unsupported medical task group"):
        MedicalTaskProtocol(group_by=("patient_name",))
    with pytest.raises(ValueError, match="Unsupported required medical tasks"):
        MedicalTaskProtocol(required_tasks=("cat_classification",))


def test_task_metrics_use_different_success_rules_by_clinical_intent():
    finding = evaluate_medical_task(
        "yes",
        {
            "medical_task": "finding_assessment",
            "references": ["yes"],
            "choices": {"A": "yes", "B": "no"},
        },
    )
    assert finding["task_success"] == 1.0

    diagnosis = evaluate_medical_task(
        "pneumonia supported by airspace opacity; no pleural effusion",
        {
            "medical_task": "diagnostic_reasoning",
            "references": ["pneumonia supported by airspace opacity"],
            "concepts": ["pneumonia"],
            "evidence": ["airspace opacity"],
            "concept_vocabulary": ["pneumonia", "pleural effusion"],
        },
    )
    assert diagnosis["concept_recall"] == 1.0
    assert diagnosis["evidence_coverage"] == 1.0
    assert diagnosis["extra_concept_rate"] == 0.0
    assert diagnosis["strict_diagnostic_accuracy"] == 1.0

    hallucinated = evaluate_medical_task(
        "pneumonia supported by airspace opacity with pleural effusion",
        {
            "medical_task": "diagnostic_reasoning",
            "references": ["pneumonia supported by airspace opacity"],
            "concepts": ["pneumonia"],
            "evidence": ["airspace opacity"],
            "concept_vocabulary": ["pneumonia", "pleural effusion"],
        },
    )
    assert hallucinated["extra_concept_rate"] == 0.5
    assert hallucinated["strict_diagnostic_accuracy"] == 0.0

    diagnosis_only = evaluate_medical_task(
        "pneumonia",
        {
            "medical_task": "diagnostic_reasoning",
            "references": ["pneumonia"],
            "concepts": ["pneumonia"],
            "concept_vocabulary": ["pneumonia"],
        },
    )
    assert diagnosis_only["strict_diagnostic_accuracy"] == 1.0
    assert diagnosis_only["task_success"] == 0.0

    report = evaluate_medical_task(
        "Findings: opacity. Impression: pneumonia.",
        {
            "medical_task": "report_generation",
            "references": ["opacity and pneumonia"],
            "concepts": ["opacity", "pneumonia"],
            "concept_vocabulary": ["opacity", "pneumonia", "effusion"],
        },
    )
    assert report["exact_match"] == 0.0
    assert report["task_success"] == 1.0


def test_task_summary_reports_macro_tasks_and_two_uncertainty_methods():
    rows = []
    for task, success in (("finding_assessment", 1.0), ("diagnostic_reasoning", 0.0)):
        rows.append(
            {
                "medical_task": task,
                "task_family": "perception" if task.startswith("finding") else "reasoning",
                "specialty": "radiology",
                "modality": "xray",
                "task_success": success,
                "strict_diagnostic_accuracy": success
                if task == "diagnostic_reasoning"
                else None,
                "exact_match": success,
                "token_f1": success,
                "concept_precision": 1.0,
                "concept_recall": success,
                "concept_f1": success,
                "evidence_coverage": success,
                "extra_concept_rate": 0.0,
                "abstained": False,
            }
        )
    summary = summarize_medical_tasks(
        rows,
        group_by=("medical_task", "task_family"),
        bootstrap_samples=50,
        confidence_level=0.95,
        seed=3,
    )
    assert summary["macro_task_success"] == 50.0
    assert summary["uncertainty"]["task_success"]["method"] == "wilson"
    assert summary["uncertainty"]["concept_recall"]["method"] == "bootstrap"
    assert set(summary["by_medical_task"]) == {
        "diagnostic_reasoning",
        "finding_assessment",
    }


def test_task_audit_distinguishes_heuristic_mapping_from_reference_source(tmp_path):
    samples = load_medical_tasks(_data_config(), project_root=PROJECT_ROOT)
    samples[0].metadata["task_mapping"] = {"method": "heuristic"}
    adapter = MedicalTasksDatasetAdapter()
    audit = audit_medical_task_dataset(
        samples,
        data_config=_data_config(),
        project_root=PROJECT_ROOT,
        dataset_fingerprint=adapter.fingerprint(_data_config(), PROJECT_ROOT),
        protocol=MedicalTaskProtocol(
            required_tasks=tuple(task.value for task in MedicalTaskType),
            require_provenance=True,
            require_deidentified=True,
        ),
    )
    assert audit["status"] == "warning"
    assert audit["mapping"] == {"heuristic_count": 1, "expert_or_native_count": 7}
    assert audit["reference_provenance"]["present"] == 8
    assert audit["samples_without_case_id"] == 8


def test_open_localization_uses_concept_coverage_not_sentence_equality():
    result = evaluate_medical_task(
        "The lesion is located in the right lung.",
        {
            "medical_task": "anatomy_localization",
            "references": ["right lung"],
            "concepts": ["right lung"],
            "concept_vocabulary": ["right lung", "left lung"],
        },
    )
    assert result["exact_match"] == 0.0
    assert result["task_success"] == 1.0


def test_task_benchmark_runs_all_eight_tasks_through_unified_runner(tmp_path):
    runtime = RuntimeContext.create(
        command="evaluation",
        config_path=PROJECT_ROOT / "pyproject.toml",
        output_directory=tmp_path,
    )
    result = MedicalTasksBenchmark().run(
        {
            "data": _data_config(),
            "model": {
                "backbone": "medical_reference",
                "parameters": {"fixed_answer": "yes"},
            },
            "output_directory": str(tmp_path),
            "protocol": {
                "required_tasks": [task.value for task in MedicalTaskType],
                "bootstrap_samples": 20,
            },
            "mode": "full",
            "batch_size": 4,
            "resume": False,
        },
        config_path=PROJECT_ROOT / "pyproject.toml",
        runtime=runtime,
    )
    score = json.loads((tmp_path / "score.json").read_text())
    assert result.dataset_size == 8
    assert result.benchmark == "medical_tasks"
    assert len(score["metrics"]["by_medical_task"]) == 8
    assert score["metadata"]["medical_task_taxonomy"] == "1.0"


def test_task_audit_fails_when_reference_provenance_is_required():
    samples = load_medical_tasks(_data_config(), project_root=PROJECT_ROOT)
    samples[0].reference_provenance = {}
    audit = audit_medical_task_dataset(
        samples,
        data_config=_data_config(),
        project_root=PROJECT_ROOT,
        dataset_fingerprint="test",
        protocol=MedicalTaskProtocol(require_reference_provenance=True),
    )
    assert audit["status"] == "failed"
    assert "lack reference provenance" in audit["errors"][0]
