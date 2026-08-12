from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medumm.core.contracts import EvaluationMode
from medumm.core.io import ensure_directory, read_jsonl, write_json, write_jsonl
from medumm.core.results import Artifact, EvaluationResult, InferenceResult


@dataclass(slots=True)
class EvaluationItem:
    sample_id: str
    request: dict[str, Any]
    content: dict[str, Any]


class EvaluationRunner:
    """Benchmark-neutral generate/score/full evaluation state machine."""

    def __init__(
        self,
        *,
        benchmark: str,
        pipeline: Any | None,
        output_directory: str | Path,
        parser: Callable[[InferenceResult], str],
        scorer: Callable[[str, dict[str, Any]], dict[str, Any]],
        summarizer: Callable[[list[dict[str, Any]]], dict[str, Any]],
        mode: EvaluationMode | str = EvaluationMode.FULL,
        resume: bool = True,
        fingerprint: str,
        metadata: dict[str, Any] | None = None,
        batch_size: int = 1,
    ) -> None:
        self.benchmark = benchmark
        self.pipeline = pipeline
        self.output_directory = ensure_directory(output_directory)
        self.parser = parser
        self.scorer = scorer
        self.summarizer = summarizer
        self.mode = EvaluationMode(mode)
        self.resume = resume
        self.fingerprint = fingerprint
        self.metadata = dict(metadata or {})
        self.batch_size = batch_size
        self.predictions_path = self.output_directory / "predictions.jsonl"

    def _previous_predictions(self, *, required: bool = False) -> dict[str, dict[str, Any]]:
        if not self.predictions_path.is_file():
            if required:
                raise FileNotFoundError(
                    f"Score mode requires predictions: {self.predictions_path}."
                )
            return {}
        if not self.resume and not required:
            return {}
        return {
            str(row["id"]): row
            for row in read_jsonl(self.predictions_path)
            if row.get("fingerprint") == self.fingerprint
        }

    def _generate(self, items: list[EvaluationItem]) -> dict[str, dict[str, Any]]:
        if self.pipeline is None:
            raise ValueError("Generate mode requires an inference pipeline.")
        predictions = self._previous_predictions()
        active = {item.sample_id for item in items}
        predictions = {key: row for key, row in predictions.items() if key in active}
        pending = [item for item in items if item.sample_id not in predictions]
        if pending:
            outputs = self.pipeline.run_many(
                [item.request for item in pending],
                batch_size=self.batch_size,
            )
            for item, output in zip(pending, outputs, strict=True):
                predictions[item.sample_id] = {
                    "id": item.sample_id,
                    "request_id": output.request_id,
                    "prediction": self.parser(output),
                    "fingerprint": self.fingerprint,
                    "model_name": output.model_name,
                }
                write_jsonl(
                    self.predictions_path,
                    [predictions[current.sample_id] for current in items if current.sample_id in predictions],
                )
        return predictions

    def _score(
        self,
        items: list[EvaluationItem],
        predictions: dict[str, dict[str, Any]],
    ) -> EvaluationResult:
        missing = [item.sample_id for item in items if item.sample_id not in predictions]
        if missing:
            raise FileNotFoundError(
                f"Missing predictions for {len(missing)} sample(s): {missing[:3]}."
            )
        rows = []
        for item in items:
            prediction = str(predictions[item.sample_id]["prediction"])
            rows.append({
                "id": item.sample_id,
                "prediction": prediction,
                **item.content,
                **self.scorer(prediction, item.content),
            })
        results_path = write_jsonl(self.output_directory / "results.jsonl", rows)
        metrics = self.summarizer(rows)
        report = {
            "schema_version": "1.0",
            "benchmark": self.benchmark,
            "mode": self.mode.value,
            "dataset_size": len(rows),
            "fingerprint": self.fingerprint,
            "metrics": metrics,
            "metadata": self.metadata,
        }
        report_path = write_json(self.output_directory / "score.json", report)
        metrics_path = self._write_metrics(metrics)
        return EvaluationResult(
            benchmark=self.benchmark,
            mode=self.mode.value,
            status="completed",
            dataset_size=len(rows),
            output_directory=str(self.output_directory),
            metrics=metrics,
            artifacts=[
                Artifact("predictions", str(self.predictions_path), "application/jsonl"),
                Artifact("results", str(results_path), "application/jsonl"),
                Artifact("report", str(report_path), "application/json"),
                Artifact("metrics", str(metrics_path), "text/csv"),
            ],
            metadata={"fingerprint": self.fingerprint, **self.metadata},
        )

    def _write_metrics(self, metrics: dict[str, Any]) -> Path:
        path = self.output_directory / "metrics.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["section", "group", "metric", "value"])
            for section, groups in metrics.items():
                groups = {"overall": groups} if isinstance(groups, dict) and "total" in groups else groups
                if not isinstance(groups, dict):
                    continue
                for group, group_metrics in groups.items():
                    if not isinstance(group_metrics, dict):
                        continue
                    for metric, value in group_metrics.items():
                        writer.writerow([section, group, metric, value])
        return path

    def run(self, items: list[EvaluationItem]) -> EvaluationResult:
        if not items:
            raise ValueError("Evaluation requires at least one item.")
        identifiers = [item.sample_id for item in items]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Evaluation item identifiers must be unique.")
        if self.mode in {EvaluationMode.GENERATE, EvaluationMode.FULL}:
            predictions = self._generate(items)
        else:
            predictions = self._previous_predictions(required=True)
        if self.mode is EvaluationMode.GENERATE:
            return EvaluationResult(
                benchmark=self.benchmark,
                mode=self.mode.value,
                status="generated",
                dataset_size=len(items),
                output_directory=str(self.output_directory),
                artifacts=[Artifact("predictions", str(self.predictions_path), "application/jsonl")],
                metadata={"fingerprint": self.fingerprint, **self.metadata},
            )
        return self._score(items, predictions)
