import json

import pytest

from medumm.core.contracts import EvaluationMode, TaskType
from medumm.core.results import InferenceResult
from medumm.evaluation.metrics import create_metric_suite, metric_suites
from medumm.evaluation.runner import EvaluationItem, EvaluationRunner
from medumm.medical.clinical_metrics import (
    evaluate_calibration,
    evaluate_grounding,
    evaluate_measurement,
    evaluate_report,
)


def test_specialized_metric_suites_are_independently_registered():
    expected = {
        "pathology_vqa",
        "medical_report_factuality",
        "medical_grounding",
        "medical_measurement",
        "medical_calibration",
    }
    for name in expected:
        assert create_metric_suite(name).version == "1.0"
    assert expected <= set(metric_suites.names())


def test_pathology_vqa_separates_binary_and_free_form_accuracy():
    suite = create_metric_suite("pathology_vqa")
    rows = []
    for prediction, content in (
        ("yes", {"references": ["yes"], "choices": {"A": "yes", "B": "no"}, "answer_type": "closed"}),
        ("no", {"references": ["yes"], "choices": {"A": "yes", "B": "no"}, "answer_type": "closed"}),
        ("canals of hering", {"references": ["canals of hering"], "choices": {}, "answer_type": "open"}),
    ):
        rows.append({**content, **suite.score(prediction, content)})
    summary = suite.summarize(rows, {"group_by": [], "bootstrap_samples": 0})
    assert summary["pathology"]["yes_no_accuracy"] == 50.0
    assert summary["pathology"]["free_form_accuracy"] == 100.0
    assert summary["pathology"]["overall_accuracy"] == 66.67
    assert summary["pathology"]["macro_answer_type_accuracy"] == 75.0


def test_report_factuality_distinguishes_negation_and_hallucination():
    result = evaluate_report(
        "Findings: right opacity and cardiomegaly. No pleural effusion. Impression: pneumonia.",
        {
            "annotations": {
                "report": {
                    "positive_findings": ["opacity", "pneumonia"],
                    "negative_findings": ["pleural effusion"],
                    "critical_findings": ["pneumonia"],
                    "finding_vocabulary": ["opacity", "pneumonia", "pleural effusion", "cardiomegaly"],
                    "required_sections": ["findings", "impression"],
                }
            }
        },
    )
    assert result["report_fact_recall"] == 1.0
    assert result["report_fact_precision"] == pytest.approx(2 / 3)
    assert result["report_contradiction_rate"] == 0.0
    assert result["report_critical_recall"] == 1.0
    assert result["report_section_completeness"] == 1.0
    assert evaluate_report("normal", {}) == {"report_available": False}


def test_grounding_handles_normalized_and_pixel_boxes_and_points():
    result = evaluate_grounding(
        json.dumps({"boxes": [[10, 10, 30, 30]], "points": [[20, 20]]}),
        {
            "annotations": {
                "grounding": {
                    "boxes": [[10, 10, 30, 30]],
                    "points": [[20, 20]],
                    "image_size": [100, 100],
                }
            }
        },
    )
    assert result["grounding_mean_iou"] == 1.0
    assert result["grounding_iou_50_recall"] == 1.0
    assert result["grounding_normalized_point_distance"] == 0.0
    assert result["grounding_pointing_accuracy"] == 1.0
    with pytest.raises(ValueError, match="image_size"):
        evaluate_grounding('{"box": [10, 10, 20, 20]}', {"annotations": {"grounding": {"boxes": [[10, 10, 20, 20]]}}})


def test_measurement_is_unit_aware_and_respects_annotation_tolerance():
    result = evaluate_measurement(
        "1.2 cm",
        {
            "annotations": {
                "measurements": [
                    {"name": "lesion diameter", "value": 12, "unit": "mm", "absolute_tolerance": 1}
                ]
            }
        },
    )
    assert result["measurement_mae"] == 0.0
    assert result["measurement_mre"] == 0.0
    assert result["measurement_within_tolerance"] == 1.0


def test_calibration_uses_full_choice_probabilities_and_selective_thresholds():
    suite = create_metric_suite("medical_calibration")
    content = {
        "references": ["pneumonia"],
        "choices": {"A": "normal", "B": "pneumonia"},
        "model_scores": {"normal": 0.1, "pneumonia": 0.9},
    }
    row = {**content, **evaluate_calibration("pneumonia", content), "modality": "xray"}
    summary = suite.summarize(
        [row],
        {
            "group_by": ["modality"],
            "calibration_bins": 10,
            "selective_thresholds": [0.5, 0.95],
            "minimum_group_samples": 1,
        },
    )
    assert row["calibration_brier"] == pytest.approx(0.02)
    assert summary["overall"]["expected_calibration_error"] == 10.0
    assert summary["overall"]["selective_prediction"]["0.5"]["coverage"] == 100.0
    assert summary["overall"]["selective_prediction"]["0.95"]["coverage"] == 0.0
    assert summary["subgroup_disparity"]["modality"]["max_min_gap"] == 0.0


class _ConfidencePipeline:
    def run_many(self, requests, batch_size=1):
        return [
            InferenceResult(
                request_id=request["request_id"],
                task=TaskType.UNDERSTANDING,
                model_name="confidence-test",
                text="yes",
                scores={"yes": 0.8, "no": 0.2},
            )
            for request in requests
        ]


def test_runner_preserves_model_scores_for_calibration(tmp_path):
    suite = create_metric_suite("medical_calibration")
    item = EvaluationItem(
        sample_id="one",
        request={"request_id": "one"},
        content={"references": ["yes"], "choices": {"A": "yes", "B": "no"}},
    )
    runner = EvaluationRunner(
        benchmark="calibration-test",
        pipeline=_ConfidencePipeline(),
        output_directory=tmp_path,
        parser=lambda output: str(output.text),
        scorer=suite.score,
        summarizer=lambda rows: suite.summarize(rows, {"group_by": []}),
        mode=EvaluationMode.FULL,
        fingerprint="test",
        resume=False,
    )
    runner.run([item])
    row = json.loads((tmp_path / "results.jsonl").read_text().strip())
    assert row["model_scores"] == {"yes": 0.8, "no": 0.2}
    assert row["calibration_available"] is True
