from __future__ import annotations

from typing import Any

from medumm.core.interfaces import MetricSuite
from medumm.core.registry import TypedRegistry
from medumm.medical.metrics import evaluate_answer, summarize_scores
from medumm.medical.task_metrics import evaluate_medical_task, summarize_medical_tasks
from medumm.medical.clinical_metrics import (
    evaluate_calibration,
    evaluate_grounding,
    evaluate_measurement,
    evaluate_report,
    summarize_calibration,
    summarize_grounding,
    summarize_groups,
    summarize_measurement,
    summarize_report,
)
from medumm.medical.benchmark_metrics import (
    evaluate_classification,
    evaluate_fairness,
    evaluate_mcqa,
    evaluate_multilabel,
    evaluate_retrieval,
    evaluate_robustness,
    evaluate_safety,
    evaluate_temporal,
    summarize_classification,
    summarize_fairness,
    summarize_mcqa,
    summarize_multilabel,
    summarize_retrieval,
    summarize_robustness,
    summarize_safety,
    summarize_temporal,
)


class MedicalVQACoreMetrics(MetricSuite):
    """Rule-based medical VQA metrics with deterministic uncertainty estimates."""

    name = "medical_vqa_core"
    version = "1.0"

    def score(self, prediction: str, content: dict[str, Any]) -> dict[str, Any]:
        return evaluate_answer(
            prediction,
            list(content["references"]),
            dict(content.get("choices", {})),
        )

    def summarize(
        self,
        rows: list[dict[str, Any]],
        protocol: dict[str, Any],
    ) -> dict[str, Any]:
        return summarize_scores(
            rows,
            group_by=tuple(protocol.get("group_by", ())),
            bootstrap_samples=int(protocol.get("bootstrap_samples", 1000)),
            confidence_level=float(protocol.get("confidence_level", 0.95)),
            seed=int(protocol.get("seed", 42)),
        )


class MedicalTaskCoreMetrics(MetricSuite):
    """Task-aware medical metrics for perception, reasoning, and generation."""

    name = "medical_task_core"
    version = "1.0"

    def score(self, prediction: str, content: dict[str, Any]) -> dict[str, Any]:
        return evaluate_medical_task(prediction, content)

    def summarize(
        self,
        rows: list[dict[str, Any]],
        protocol: dict[str, Any],
    ) -> dict[str, Any]:
        return summarize_medical_tasks(
            rows,
            group_by=tuple(protocol.get("group_by", ())),
            bootstrap_samples=int(protocol.get("bootstrap_samples", 1000)),
            confidence_level=float(protocol.get("confidence_level", 0.95)),
            seed=int(protocol.get("seed", 42)),
        )


class PathologyVQAMetrics(MedicalVQACoreMetrics):
    """PathVQA-compatible yes/no, free-form, and overall answer scoring."""

    name = "pathology_vqa"
    version = "1.0"

    def summarize(
        self,
        rows: list[dict[str, Any]],
        protocol: dict[str, Any],
    ) -> dict[str, Any]:
        summary = super().summarize(rows, protocol)
        yes_no = [row for row in rows if str(row.get("answer_type")) == "closed"]
        free_form = [row for row in rows if str(row.get("answer_type")) != "closed"]

        def accuracy(values: list[dict[str, Any]]) -> float | None:
            if not values:
                return None
            return round(100 * sum(float(row["exact_match"]) for row in values) / len(values), 2)

        available = [value for value in (accuracy(yes_no), accuracy(free_form)) if value is not None]
        summary["pathology"] = {
            "total": len(rows),
            "yes_no_count": len(yes_no),
            "free_form_count": len(free_form),
            "yes_no_accuracy": accuracy(yes_no),
            "free_form_accuracy": accuracy(free_form),
            "overall_accuracy": accuracy(rows),
            "macro_answer_type_accuracy": round(sum(available) / len(available), 2)
            if available
            else None,
        }
        return summary


class MedicalReportMetrics(MetricSuite):
    """Auditable factuality and structure metrics for annotated medical reports."""

    name = "medical_report_factuality"
    version = "1.0"

    def score(self, prediction: str, content: dict[str, Any]) -> dict[str, Any]:
        return evaluate_report(prediction, content)

    def summarize(self, rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
        return summarize_groups(
            rows, protocol, summarize_report, primary_metric="fact_f1"
        )


class MedicalGroundingMetrics(MetricSuite):
    """Normalized box IoU and point localization for medical images."""

    name = "medical_grounding"
    version = "1.0"

    def score(self, prediction: str, content: dict[str, Any]) -> dict[str, Any]:
        return evaluate_grounding(prediction, content)

    def summarize(self, rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
        return summarize_groups(
            rows, protocol, summarize_grounding, primary_metric="mean_iou"
        )


class MedicalMeasurementMetrics(MetricSuite):
    """Unit-aware physical measurement error and tolerance scoring."""

    name = "medical_measurement"
    version = "1.0"

    def score(self, prediction: str, content: dict[str, Any]) -> dict[str, Any]:
        return evaluate_measurement(prediction, content)

    def summarize(self, rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
        return summarize_groups(
            rows, protocol, summarize_measurement, primary_metric="within_tolerance"
        )


class MedicalCalibrationMetrics(MetricSuite):
    """Proper scoring, ECE, and selective prediction over model probabilities."""

    name = "medical_calibration"
    version = "1.0"

    def score(self, prediction: str, content: dict[str, Any]) -> dict[str, Any]:
        return evaluate_calibration(prediction, content)

    def summarize(self, rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
        def aggregate(values: list[dict[str, Any]]) -> dict[str, Any]:
            return summarize_calibration(values, protocol)

        return summarize_groups(rows, protocol, aggregate, primary_metric="accuracy")


class MedicalMCQAMetrics(MetricSuite):
    """Strict choice extraction, accuracy, and invalid-response accounting."""

    name = "medical_mcqa"
    version = "1.0"

    def score(self, prediction: str, content: dict[str, Any]) -> dict[str, Any]:
        return evaluate_mcqa(prediction, content)

    def summarize(self, rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
        return summarize_groups(rows, protocol, summarize_mcqa, primary_metric="choice_accuracy")


class MedicalImageClassificationMetrics(MetricSuite):
    """Accuracy, balanced accuracy, macro F1, confusion, and optional macro AUC."""

    name = "medical_image_classification"
    version = "1.0"

    def score(self, prediction: str, content: dict[str, Any]) -> dict[str, Any]:
        return evaluate_classification(prediction, content)

    def summarize(self, rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
        return summarize_groups(
            rows, protocol, summarize_classification, primary_metric="balanced_accuracy"
        )


class MedicalMultilabelFindingMetrics(MetricSuite):
    """Per-sample and micro multilabel finding precision, recall, F1, and exact match."""

    name = "medical_multilabel_findings"
    version = "1.0"

    def score(self, prediction: str, content: dict[str, Any]) -> dict[str, Any]:
        return evaluate_multilabel(prediction, content)

    def summarize(self, rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
        return summarize_groups(rows, protocol, summarize_multilabel, primary_metric="micro_f1")


class MedicalTemporalReasoningMetrics(MetricSuite):
    """Phase/order accuracy, edit similarity, and transition F1 for medical sequences."""

    name = "medical_temporal_reasoning"
    version = "1.0"

    def score(self, prediction: str, content: dict[str, Any]) -> dict[str, Any]:
        return evaluate_temporal(prediction, content)

    def summarize(self, rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
        return summarize_groups(rows, protocol, summarize_temporal, primary_metric="edit_similarity")


class MedicalRetrievalMetrics(MetricSuite):
    """Recall@K and mean reciprocal rank from model-produced candidate scores."""

    name = "medical_retrieval"
    version = "1.0"

    def score(self, prediction: str, content: dict[str, Any]) -> dict[str, Any]:
        return evaluate_retrieval(prediction, content)

    def summarize(self, rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
        return summarize_groups(rows, protocol, summarize_retrieval, primary_metric="recall_at_1")


class MedicalFairnessMetrics(MetricSuite):
    """Worst-group accuracy plus demographic-parity and equal-opportunity gaps."""

    name = "medical_fairness"
    version = "1.0"

    def score(self, prediction: str, content: dict[str, Any]) -> dict[str, Any]:
        return evaluate_fairness(prediction, content)

    def summarize(self, rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
        del protocol
        return {"overall": summarize_fairness(rows)}


class MedicalSafetyMetrics(MetricSuite):
    """Expected refusal, unsafe compliance, over-refusal, and policy-term coverage."""

    name = "medical_safety"
    version = "1.0"

    def score(self, prediction: str, content: dict[str, Any]) -> dict[str, Any]:
        return evaluate_safety(prediction, content)

    def summarize(self, rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
        del protocol
        return {"overall": summarize_safety(rows)}


class MedicalRobustnessMetrics(MetricSuite):
    """Paired baseline/perturbation accuracy drop and prediction consistency."""

    name = "medical_robustness"
    version = "1.0"

    def score(self, prediction: str, content: dict[str, Any]) -> dict[str, Any]:
        return evaluate_robustness(prediction, content)

    def summarize(self, rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
        del protocol
        return {"overall": summarize_robustness(rows)}


metric_suites: TypedRegistry[MetricSuite] = TypedRegistry("metric_suite")


def register_metric_suites() -> None:
    if not metric_suites.contains(MedicalVQACoreMetrics.name):
        metric_suites.register(
            MedicalVQACoreMetrics.name,
            MedicalVQACoreMetrics,
            description="Exact match, token F1, abstention, closed-answer and uncertainty metrics",
            metadata={"version": MedicalVQACoreMetrics.version},
        )
    if not metric_suites.contains(MedicalTaskCoreMetrics.name):
        metric_suites.register(
            MedicalTaskCoreMetrics.name,
            MedicalTaskCoreMetrics,
            description=(
                "Task success, concept/evidence coverage, strict diagnosis, "
                "hallucinated concepts, and uncertainty"
            ),
            metadata={"version": MedicalTaskCoreMetrics.version},
        )
    suites = (
        (
            PathologyVQAMetrics,
            "Pathology yes/no, free-form, overall, and macro answer-type accuracy",
        ),
        (
            MedicalReportMetrics,
            "Annotated report factuality, contradiction, critical finding, and section metrics",
        ),
        (
            MedicalGroundingMetrics,
            "Normalized medical box IoU and point localization metrics",
        ),
        (
            MedicalMeasurementMetrics,
            "Unit-aware physical measurement error and tolerance metrics",
        ),
        (
            MedicalCalibrationMetrics,
            "ECE, Brier, NLL, confidence, and selective prediction metrics",
        ),
        (
            MedicalMCQAMetrics,
            "Strict medical multiple-choice accuracy and invalid-response rate",
        ),
        (
            MedicalImageClassificationMetrics,
            "Balanced accuracy, macro F1, confusion, and optional macro AUC",
        ),
        (
            MedicalMultilabelFindingMetrics,
            "Medical finding micro/macro F1 and exact multilabel match",
        ),
        (
            MedicalTemporalReasoningMetrics,
            "Medical sequence edit similarity, phase accuracy, and transition F1",
        ),
        (
            MedicalRetrievalMetrics,
            "Medical image-text retrieval Recall@K and mean reciprocal rank",
        ),
        (
            MedicalFairnessMetrics,
            "Worst-group accuracy and demographic/equal-opportunity disparity",
        ),
        (
            MedicalSafetyMetrics,
            "Expected refusal, unsafe compliance, and over-refusal metrics",
        ),
        (
            MedicalRobustnessMetrics,
            "Paired perturbation accuracy drop and response consistency",
        ),
    )
    for suite, description in suites:
        if not metric_suites.contains(suite.name):
            metric_suites.register(
                suite.name,
                suite,
                description=description,
                metadata={"version": suite.version},
            )


def create_metric_suite(name: str) -> MetricSuite:
    register_metric_suites()
    return metric_suites.create(name)
