from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medumm.core.io import ensure_directory, read_jsonl, write_json, write_jsonl


@dataclass(slots=True)
class EvaluationItem:
    sample_id: str
    request: dict[str, Any]
    content: dict[str, Any]


class EvaluationRunner:
    """Reusable generate/score runner with resumable predictions."""

    def __init__(
        self,
        *,
        benchmark: str,
        pipeline: Any,
        output_directory: str | Path,
        parser: Callable[[Any], str],
        scorer: Callable[[str, dict[str, Any]], dict[str, Any]],
        summarizer: Callable[[list[dict[str, Any]]], dict[str, Any]],
        resume: bool = True,
    ) -> None:
        self.benchmark = benchmark
        self.pipeline = pipeline
        self.output_directory = ensure_directory(output_directory)
        self.parser = parser
        self.scorer = scorer
        self.summarizer = summarizer
        self.resume = resume
        self.predictions_path = self.output_directory / "predictions.jsonl"

    def run(self, items: list[EvaluationItem]) -> dict[str, Any]:
        if not items:
            raise ValueError("Evaluation requires at least one item.")
        previous = read_jsonl(self.predictions_path) if self.resume and self.predictions_path.exists() else []
        predictions = {str(row["id"]): row for row in previous}
        active = {item.sample_id for item in items}
        predictions = {key: value for key, value in predictions.items() if key in active}
        for item in items:
            if item.sample_id not in predictions:
                predictions[item.sample_id] = {
                    "id": item.sample_id,
                    "prediction": self.parser(self.pipeline.run(item.request)),
                }
                write_jsonl(
                    self.predictions_path,
                    [predictions[current.sample_id] for current in items if current.sample_id in predictions],
                )
        results = []
        for item in items:
            prediction = str(predictions[item.sample_id]["prediction"])
            results.append({
                "id": item.sample_id,
                "prediction": prediction,
                **item.content,
                **self.scorer(prediction, item.content),
            })
        results_path = write_jsonl(self.output_directory / "results.jsonl", results)
        report = {
            "benchmark": self.benchmark,
            "dataset_size": len(results),
            "metrics": self.summarizer(results),
            "clinical_use": False,
        }
        report_path = write_json(self.output_directory / "score.json", report)
        metrics_path = self.output_directory / "metrics.csv"
        with metrics_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["section", "group", "metric", "value"])
            for section, groups in report["metrics"].items():
                groups = {"overall": groups} if "total" in groups else groups
                for group, metrics in groups.items():
                    for metric, value in metrics.items():
                        writer.writerow([section, group, metric, value])
        return {
            **report,
            "status": "completed",
            "predictions_path": str(self.predictions_path),
            "results_path": str(results_path),
            "report_path": str(report_path),
            "metrics_path": str(metrics_path),
        }
