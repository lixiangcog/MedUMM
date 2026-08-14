from __future__ import annotations

import json
from pathlib import Path

import pytest

from medumm.api import evaluate
from medumm.cli.main import main
from medumm.core.builtins import register_builtins
from medumm.core.config import load_config
from medumm.core.registry import registry
from medumm.core.runtime import RuntimeContext
from medumm.evaluation.benchmark_catalog import (
    SPECIALIZED_BENCHMARKS,
    get_medical_benchmark,
)
from medumm.evaluation.metrics import create_metric_suite
from medumm.evaluation.specialized import SpecializedMedicalBenchmark
from medumm.resources import DATASET_RESOURCES
from tests.conftest import PROJECT_ROOT


CONFIG_ROOT = PROJECT_ROOT / "configs/evaluation/benchmarks_v1.6"
BENCHMARK_NAMES = {spec.name for spec in SPECIALIZED_BENCHMARKS}


def test_specialized_catalog_is_not_dataset_catalog_in_disguise():
    register_builtins()
    registered = set(registry.benchmarks.names())
    assert len(SPECIALIZED_BENCHMARKS) == 13
    assert BENCHMARK_NAMES <= registered
    assert len(registered - {"cross_task"}) == 15
    assert len(DATASET_RESOURCES.values()) == 34
    assert all(create_metric_suite(spec.metric_suite).version == spec.version for spec in SPECIALIZED_BENCHMARKS)


def test_every_routed_dataset_uses_a_registered_compatible_benchmark():
    register_builtins()
    registered = set(registry.benchmarks.names())
    for resource in DATASET_RESOURCES.values():
        assert resource.benchmark in registered
        if resource.benchmark in BENCHMARK_NAMES:
            contract = get_medical_benchmark(resource.benchmark)
            assert resource.adapter_family in contract.dataset_families


@pytest.mark.parametrize("spec", SPECIALIZED_BENCHMARKS, ids=lambda value: value.name)
def test_every_specialized_benchmark_runs_audit_inference_score_and_report(spec, tmp_path):
    config_path = CONFIG_ROOT / f"{spec.name}.yaml"
    config = load_config(config_path)
    output = tmp_path / spec.name
    config["evaluation"]["output_directory"] = str(output)
    result = evaluate(config, config_path=config_path)
    report = json.loads((output / "score.json").read_text(encoding="utf-8"))
    audit = json.loads((output / "dataset_audit.json").read_text(encoding="utf-8"))

    assert result.status == "completed"
    assert result.dataset_size >= 1
    assert report["benchmark"] == spec.name
    assert report["metadata"]["benchmark_spec"]["metric_suite"] == spec.metric_suite
    assert report["metadata"]["protocol"]["metric_suite"] == spec.metric_suite
    assert audit["status"] == "passed"
    assert (output / "predictions.jsonl").is_file()
    assert (output / "results.jsonl").is_file()
    assert (output / "metrics.csv").is_file()


def test_specialized_metrics_cover_distinct_medical_failure_modes():
    cases = {
        "medical_mcqa": (
            "B",
            {"references": ["B"], "choices": {"A": "normal", "B": "opacity"}},
            "choice_correct",
        ),
        "medical_multilabel_findings": (
            '{"labels":["opacity","effusion"]}',
            {"annotations": {"multilabel": {"labels": ["opacity", "effusion"]}}},
            "multilabel_exact_match",
        ),
        "medical_temporal_reasoning": (
            "incision -> exposure -> closure",
            {"annotations": {"temporal": {"sequence": ["incision", "exposure", "closure"]}}},
            "temporal_exact_match",
        ),
        "medical_retrieval": (
            "",
            {
                "annotations": {
                    "retrieval": {
                        "candidates": ["normal", "opacity"],
                        "positives": ["opacity"],
                    }
                },
                "model_scores": {"normal": 0.1, "opacity": 0.9},
            },
            "retrieval_recall_at_1",
        ),
        "medical_safety": (
            "I cannot provide a dose; consult a clinician.",
            {"annotations": {"safety": {"should_refuse": True}}},
            "safety_pass",
        ),
    }
    for suite_name, (prediction, content, metric) in cases.items():
        assert create_metric_suite(suite_name).score(prediction, content)[metric] == 1.0


def test_robustness_audit_fails_closed_on_incomplete_pairs(tmp_path):
    manifest = tmp_path / "incomplete.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "only-baseline",
                "task": "finding_assessment",
                "prompt": "Is a focus present?",
                "references": ["A"],
                "choices": {"A": "yes", "B": "no"},
                "annotations": {
                    "robustness": {"pair_id": "p1", "variant": "baseline"}
                },
                "metadata": {"benchmark": "medical_robustness"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    runtime = RuntimeContext.create(
        command="evaluation",
        config_path=PROJECT_ROOT / "pyproject.toml",
        output_directory=tmp_path / "output",
    )
    with pytest.raises(ValueError, match="Robustness pairs are incomplete"):
        SpecializedMedicalBenchmark("medical_robustness").run(
            {
                "data": {
                    "adapter": "medical_tasks_jsonl",
                    "path": str(manifest),
                    "benchmark_filter": "medical_robustness",
                    "validate_media": True,
                    "deidentified": True,
                },
                "output_directory": str(tmp_path / "output"),
                "protocol": {"require_complete_pairs": True},
                "mode": "audit",
            },
            config_path=PROJECT_ROOT / "pyproject.toml",
            runtime=runtime,
        )


def test_benchmark_cli_reports_truthful_counts_and_contracts(capsys):
    assert main(["benchmarks", "audit"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] is True
    assert report["specialized_medical_benchmarks"] == 13
    assert report["independent_benchmarks"] == 15
    assert report["dataset_resources"] == 34
    assert "not counted as benchmarks" in report["validation_scope"]

    assert main(["benchmarks", "show", "medical_grounding"]) == 0
    contract = json.loads(capsys.readouterr().out)
    assert contract["required_annotation"] == "grounding"
    assert contract["metric_suite"] == "medical_grounding"

    assert main(["benchmarks", "template", "medical_safety"]) == 0
    template = capsys.readouterr().out
    assert "required_annotation: safety" in template
    assert "metric_suite: medical_safety" in template
