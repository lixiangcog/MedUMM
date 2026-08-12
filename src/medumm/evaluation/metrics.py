from __future__ import annotations

from typing import Any

from medumm.core.interfaces import MetricSuite
from medumm.core.registry import TypedRegistry
from medumm.medical.metrics import evaluate_answer, summarize_scores


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


metric_suites: TypedRegistry[MetricSuite] = TypedRegistry("metric_suite")


def register_metric_suites() -> None:
    if not metric_suites.contains(MedicalVQACoreMetrics.name):
        metric_suites.register(
            MedicalVQACoreMetrics.name,
            MedicalVQACoreMetrics,
            description="Exact match, token F1, abstention, closed-answer and uncertainty metrics",
            metadata={"version": MedicalVQACoreMetrics.version},
        )


def create_metric_suite(name: str) -> MetricSuite:
    register_metric_suites()
    return metric_suites.create(name)
