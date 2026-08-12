from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from medumm.core.builtins import register_builtins
from medumm.core.config import execution_config
from medumm.core.contracts import EvaluationMode
from medumm.core.interfaces import BenchmarkAdapter
from medumm.core.io import ensure_directory, write_json
from medumm.core.registry import registry
from medumm.core.results import Artifact, EvaluationResult, InferenceResult
from medumm.core.runtime import RuntimeContext
from medumm.evaluation.runner import EvaluationItem, EvaluationRunner
from medumm.inference import InferencePipeline
from medumm.evaluation.metrics import create_metric_suite
from medumm.evaluation.protocol import EvaluationProtocol, audit_medical_vqa_dataset


def _parse_output(output: InferenceResult) -> str:
    return str(output.text or "")


class MedicalVQABenchmark(BenchmarkAdapter):
    name = "medical_vqa"

    def run(
        self,
        config: dict[str, Any],
        *,
        config_path: str | Path,
        runtime: RuntimeContext,
    ) -> EvaluationResult:
        register_builtins()
        data_config = config.get("data")
        evaluation_config = config
        if not isinstance(data_config, dict):
            raise ValueError("Evaluation config requires a data mapping.")
        mode = EvaluationMode(evaluation_config.get("mode", "full"))
        model_config = config.get("model", {})
        if mode is not EvaluationMode.AUDIT and not isinstance(model_config, dict):
            raise ValueError("Evaluation config requires a model mapping.")
        if not isinstance(model_config, dict):
            model_config = {}
        dataset_name = str(data_config.get("adapter", "medical_vqa_jsonl"))
        dataset = registry.datasets.create(dataset_name)
        samples = dataset.load(data_config, runtime.project_root)
        dataset_fingerprint = dataset.fingerprint(data_config, runtime.project_root)
        raw_protocol = evaluation_config.get("protocol")
        if raw_protocol is not None and not isinstance(raw_protocol, dict):
            raise ValueError("Evaluation protocol must be a mapping when provided.")
        metric_suite_name = str(
            (raw_protocol or {}).get("metric_suite", "medical_vqa_core")
        ).strip().lower()
        metric_suite = create_metric_suite(metric_suite_name)
        protocol = EvaluationProtocol.from_config(
            raw_protocol,
            seed=runtime.seed,
            metric_suite_version=metric_suite.version,
        )
        if protocol.metric_suite != metric_suite.name:
            raise ValueError("Resolved metric suite does not match the evaluation protocol.")
        output_directory = Path(
            evaluation_config.get("output_directory", "outputs/evaluation/medical_vqa")
        )
        output_directory = (
            output_directory
            if output_directory.is_absolute()
            else runtime.project_root / output_directory
        )
        ensure_directory(output_directory)
        audit = audit_medical_vqa_dataset(
            samples,
            data_config=data_config,
            project_root=runtime.project_root,
            dataset_fingerprint=dataset_fingerprint,
            protocol=protocol,
        )
        audit_path = write_json(output_directory / "dataset_audit.json", audit)
        if audit["status"] == "failed":
            raise ValueError(
                "Medical dataset audit failed: " + "; ".join(audit["errors"])
            )
        if mode is EvaluationMode.AUDIT:
            return EvaluationResult(
                benchmark=self.name,
                mode=mode.value,
                status=str(audit["status"]),
                dataset_size=len(samples),
                output_directory=str(output_directory),
                metrics={"data_quality": audit},
                artifacts=[Artifact("dataset_audit", str(audit_path), "application/json")],
                metadata={
                    "dataset": dataset_name,
                    "dataset_fingerprint": dataset_fingerprint,
                    "protocol": protocol.to_dict(),
                    "clinical_use": False,
                },
            )
        backbone = str(model_config.get("backbone", "")).strip().lower()
        if not backbone:
            raise ValueError("Evaluation model requires a backbone.")
        model_identity = {
            "backbone": backbone,
            "config": model_config.get("config", {}),
            "parameters": model_config.get("parameters", {}),
            "prompt_template": evaluation_config.get("prompt_template", "{question}"),
        }
        run_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "dataset": dataset_fingerprint,
                    "model": model_identity,
                    "protocol": protocol.to_dict(),
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        prompt_template = str(evaluation_config.get("prompt_template", "{question}"))
        if "{question}" not in prompt_template:
            raise ValueError("Evaluation prompt_template must contain {question}.")
        items = [
            EvaluationItem(
                sample_id=sample.sample_id,
                request={
                    "request_id": sample.sample_id,
                    "task": "understanding",
                    "prompt": prompt_template.format(question=sample.question),
                    "images": sample.image_paths,
                    "parameters": dict(model_config.get("parameters", {})),
                },
                content={
                    "references": sample.answers,
                    "choices": sample.choices,
                    "answer_type": sample.answer_type,
                    "modality": sample.modality,
                    "category": sample.category,
                    "language": sample.language,
                    "sample_metadata": sample.metadata,
                },
            )
            for sample in samples
        ]
        pipeline = None
        if mode is not EvaluationMode.SCORE:
            pipeline = InferencePipeline(
                backbone,
                dict(model_config.get("config", {})),
                runtime=runtime,
            )
        try:
            raw_shard = evaluation_config.get("shard", {})
            if not isinstance(raw_shard, dict):
                raise ValueError("Evaluation shard must be a mapping when provided.")
            shard_count = int(
                raw_shard.get("count", runtime.world_size if runtime.distributed else 1)
            )
            shard_rank = int(
                raw_shard.get("rank", runtime.rank if shard_count > 1 else 0)
            )
            runner = EvaluationRunner(
                benchmark=self.name,
                pipeline=pipeline,
                output_directory=output_directory,
                parser=_parse_output,
                scorer=metric_suite.score,
                summarizer=lambda rows: metric_suite.summarize(
                    rows, protocol.to_dict()
                ),
                mode=mode,
                resume=bool(evaluation_config.get("resume", True)),
                fingerprint=run_fingerprint,
                metadata={
                    "dataset": dataset_name,
                    "dataset_fingerprint": dataset_fingerprint,
                    "model": backbone,
                    "protocol": protocol.to_dict(),
                    "dataset_audit": {
                        "status": audit["status"],
                        "warnings": len(audit["warnings"]),
                    },
                    "clinical_use": False,
                },
                batch_size=int(evaluation_config.get("batch_size", 1)),
                shard_rank=shard_rank,
                shard_count=shard_count,
                prediction_flush_interval=int(
                    evaluation_config.get("checkpoint", {}).get(
                        "flush_every", evaluation_config.get("batch_size", 1)
                    )
                ),
            )
            shard_items = items[runner.shard_rank :: runner.shard_count]
            if not shard_items:
                raise ValueError(
                    f"Evaluation shard {runner.shard_rank} has no samples."
                )
            result = runner.run(shard_items)
            result.artifacts.append(
                Artifact("dataset_audit", str(audit_path), "application/json")
            )
            return result
        finally:
            if pipeline is not None:
                pipeline.close()


def run_medical_vqa(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    runtime: RuntimeContext | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper retained for v0.1 Python callers."""

    context = runtime or RuntimeContext.create(
        command="evaluation",
        config_path=config_path,
        output_directory=execution_config(config, "evaluation").get("output_directory"),
        runtime_config=config.get("runtime"),
    )
    return MedicalVQABenchmark().run(
        execution_config(config, "evaluation"),
        config_path=config_path,
        runtime=context,
    ).to_dict()
