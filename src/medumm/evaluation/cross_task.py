from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from medumm.core.builtins import register_builtins
from medumm.core.interfaces import BenchmarkAdapter
from medumm.core.io import ensure_directory, write_json
from medumm.core.registry import registry
from medumm.core.results import Artifact, EvaluationResult
from medumm.core.runtime import RuntimeContext


class CrossTaskBenchmark(BenchmarkAdapter):
    """Compose registered benchmarks without coupling their task semantics."""

    name = "cross_task"

    def run(
        self,
        config: dict[str, Any],
        *,
        config_path: str | Path,
        runtime: RuntimeContext,
    ) -> EvaluationResult:
        register_builtins()
        specifications = config.get("benchmarks")
        if not isinstance(specifications, list) or not specifications:
            raise ValueError("Cross-task evaluation requires a non-empty benchmarks list.")
        output_directory = Path(
            config.get("output_directory", "outputs/evaluation/cross_task")
        )
        if not output_directory.is_absolute():
            output_directory = runtime.project_root / output_directory
        ensure_directory(output_directory)

        results: list[EvaluationResult] = []
        seen: dict[str, int] = {}
        labels: set[str] = set()
        for index, raw_specification in enumerate(specifications):
            if not isinstance(raw_specification, dict):
                raise ValueError(f"benchmarks[{index}] must be a mapping.")
            specification = dict(raw_specification)
            benchmark_name = str(
                specification.get("benchmark", "medical_vqa")
            ).strip().lower()
            if benchmark_name == self.name:
                raise ValueError("Cross-task evaluation cannot contain itself.")
            seen[benchmark_name] = seen.get(benchmark_name, 0) + 1
            label = str(specification.pop("name", "")).strip()
            if not label:
                suffix = f"-{seen[benchmark_name]}" if seen[benchmark_name] > 1 else ""
                label = f"{benchmark_name}{suffix}"
            if label in labels:
                raise ValueError(f"Cross-task benchmark name {label!r} is duplicated.")
            labels.add(label)
            specification["benchmark"] = benchmark_name
            child_output = specification.get("output_directory", output_directory / label)
            child_output = Path(child_output)
            if not child_output.is_absolute():
                child_output = runtime.project_root / child_output
            specification["output_directory"] = str(child_output)
            child_runtime = replace(
                runtime,
                output_directory=ensure_directory(child_output),
                metadata={**runtime.metadata, "parent_run_id": runtime.run_id, "benchmark": label},
            )
            benchmark = registry.benchmarks.create(benchmark_name)
            results.append(
                benchmark.run(
                    specification,
                    config_path=config_path,
                    runtime=child_runtime,
                )
            )

        completed = sum(result.status in {"completed", "generated"} for result in results)
        total_samples = sum(result.dataset_size for result in results)
        report = {
            "schema_version": "1.0",
            "benchmark": self.name,
            "status": "completed" if completed == len(results) else "partial",
            "benchmark_count": len(results),
            "completed": completed,
            "dataset_size": total_samples,
            "results": [result.to_dict() for result in results],
        }
        report_path = write_json(output_directory / "cross_task_report.json", report)
        return EvaluationResult(
            benchmark=self.name,
            mode="composite",
            status=str(report["status"]),
            dataset_size=total_samples,
            output_directory=str(output_directory),
            metrics={
                "summary": {
                    "benchmark_count": len(results),
                    "completed": completed,
                    "dataset_size": total_samples,
                }
            },
            artifacts=[Artifact("cross_task_report", str(report_path), "application/json")],
            metadata={"benchmarks": [result.benchmark for result in results]},
        )
