from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from medumm.core.builtins import register_builtins
from medumm.core.contracts import EvaluationMode
from medumm.core.interfaces import BenchmarkAdapter
from medumm.core.io import ensure_directory, write_json
from medumm.core.registry import registry
from medumm.core.results import Artifact, EvaluationResult, InferenceResult
from medumm.core.runtime import RuntimeContext
from medumm.evaluation.medical_task_protocol import (
    MedicalTaskProtocol,
    audit_medical_task_dataset,
)
from medumm.evaluation.metrics import create_metric_suite
from medumm.evaluation.runner import EvaluationItem, EvaluationRunner
from medumm.inference import InferencePipeline


DEFAULT_PROMPTS = {
    "finding_assessment": "{prompt}\nAnswer yes or no only.",
    "clinical_description": "{prompt}\nDescribe the visible medical findings concisely.",
    "anatomy_localization": "{prompt}\nName the anatomical location concisely.",
    "quantitative_assessment": "{prompt}\nGive the requested measurement or count concisely.",
    "image_context": "{prompt}\nIdentify the imaging context or acquisition information concisely.",
    "diagnostic_reasoning": (
        "{prompt}\nGive the most likely diagnosis and the image evidence supporting it."
    ),
    "report_generation": (
        "{prompt}\nWrite a structured findings and impression response using only visible evidence."
    ),
    "patient_communication": (
        "{prompt}\nExplain the result in plain language and state uncertainty; do not add unsupported facts."
    ),
}


def _parse_output(output: InferenceResult) -> str:
    return str(output.text or "")


class MedicalTasksBenchmark(BenchmarkAdapter):
    """Evaluate clinical intents without reducing them to image classes."""

    name = "medical_tasks"

    def run(
        self,
        config: dict[str, Any],
        *,
        config_path: str | Path,
        runtime: RuntimeContext,
    ) -> EvaluationResult:
        register_builtins()
        data_config = config.get("data")
        if not isinstance(data_config, dict):
            raise ValueError("Medical task evaluation requires a data mapping.")
        mode = EvaluationMode(config.get("mode", "full"))
        model_config = config.get("model", {})
        if mode is not EvaluationMode.AUDIT and not isinstance(model_config, dict):
            raise ValueError("Medical task evaluation requires a model mapping.")
        if not isinstance(model_config, dict):
            model_config = {}

        dataset_name = str(data_config.get("adapter", "medical_tasks_jsonl"))
        dataset = registry.datasets.create(dataset_name)
        samples = dataset.load(data_config, runtime.project_root)
        dataset_fingerprint = dataset.fingerprint(data_config, runtime.project_root)
        raw_protocol = config.get("protocol")
        if raw_protocol is not None and not isinstance(raw_protocol, dict):
            raise ValueError("Medical task protocol must be a mapping when provided.")
        metric_suite = create_metric_suite(
            str((raw_protocol or {}).get("metric_suite", "medical_task_core"))
        )
        protocol = MedicalTaskProtocol.from_config(
            raw_protocol,
            seed=runtime.seed,
            metric_suite_version=metric_suite.version,
        )
        if protocol.metric_suite != metric_suite.name:
            raise ValueError("Resolved metric suite does not match the medical task protocol.")

        output_directory = Path(
            config.get("output_directory", "outputs/evaluation/medical_tasks")
        )
        output_directory = (
            output_directory
            if output_directory.is_absolute()
            else runtime.project_root / output_directory
        )
        ensure_directory(output_directory)
        audit = audit_medical_task_dataset(
            samples,
            data_config=data_config,
            project_root=runtime.project_root,
            dataset_fingerprint=dataset_fingerprint,
            protocol=protocol,
        )
        audit_path = write_json(output_directory / "dataset_audit.json", audit)
        if audit["status"] == "failed":
            raise ValueError("Medical task dataset audit failed: " + "; ".join(audit["errors"]))
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
            raise ValueError("Medical task evaluation model requires a backbone.")
        raw_prompts = config.get("task_prompts", {})
        if raw_prompts is not None and not isinstance(raw_prompts, dict):
            raise ValueError("evaluation.task_prompts must be a mapping.")
        prompt_templates = {**DEFAULT_PROMPTS, **dict(raw_prompts or {})}
        for task, template in prompt_templates.items():
            if "{prompt}" not in str(template):
                raise ValueError(f"Prompt template for {task} must contain {{prompt}}.")

        concept_vocabularies = {
            task: sorted(
                {
                    concept
                    for sample in samples
                    if sample.task.value == task
                    for concept in sample.concepts
                    if concept.strip()
                }
            )
            for task in {sample.task.value for sample in samples}
        }
        model_identity = {
            "backbone": backbone,
            "config": model_config.get("config", {}),
            "parameters": model_config.get("parameters", {}),
            "task_prompts": prompt_templates,
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
        items = [
            EvaluationItem(
                sample_id=sample.sample_id,
                request={
                    "request_id": sample.sample_id,
                    "task": "understanding",
                    "medical_task": sample.task.value,
                    "prompt": str(prompt_templates[sample.task.value]).format(
                        prompt=sample.prompt
                    ),
                    "images": sample.image_paths,
                    "volumes": sample.volume_paths,
                    "videos": sample.video_paths,
                    "parameters": dict(model_config.get("parameters", {})),
                    "metadata": {
                        "medical_task": sample.task.value,
                        "task_family": sample.task_family,
                        "specialty": sample.specialty,
                        "modality": sample.modality,
                    },
                },
                content={
                    "references": sample.references,
                    "choices": sample.choices,
                    "medical_task": sample.task.value,
                    "task_family": sample.task_family,
                    "specialty": sample.specialty,
                    "modality": sample.modality,
                    "anatomy": sample.anatomy,
                    "answer_type": sample.answer_type,
                    "language": sample.language,
                    "concepts": sample.concepts,
                    "evidence": sample.evidence,
                    "concept_vocabulary": concept_vocabularies[sample.task.value],
                    "case_id": sample.case_id,
                    "turn_index": sample.turn_index,
                    "reference_provenance": sample.reference_provenance,
                    "sample_metadata": sample.metadata,
                },
            )
            for sample in samples
        ]

        pipeline = None
        if mode is not EvaluationMode.SCORE:
            pipeline = InferencePipeline(
                backbone, dict(model_config.get("config", {})), runtime=runtime
            )
        try:
            raw_shard = config.get("shard", {})
            if not isinstance(raw_shard, dict):
                raise ValueError("Evaluation shard must be a mapping when provided.")
            shard_count = int(
                raw_shard.get("count", runtime.world_size if runtime.distributed else 1)
            )
            shard_rank = int(raw_shard.get("rank", runtime.rank if shard_count > 1 else 0))
            runner = EvaluationRunner(
                benchmark=self.name,
                pipeline=pipeline,
                output_directory=output_directory,
                parser=_parse_output,
                scorer=metric_suite.score,
                summarizer=lambda rows: metric_suite.summarize(rows, protocol.to_dict()),
                mode=mode,
                resume=bool(config.get("resume", True)),
                fingerprint=run_fingerprint,
                metadata={
                    "dataset": dataset_name,
                    "dataset_fingerprint": dataset_fingerprint,
                    "model": backbone,
                    "protocol": protocol.to_dict(),
                    "medical_task_taxonomy": "1.0",
                    "dataset_audit": {
                        "status": audit["status"],
                        "warnings": len(audit["warnings"]),
                    },
                    "clinical_use": False,
                },
                batch_size=int(config.get("batch_size", 1)),
                shard_rank=shard_rank,
                shard_count=shard_count,
                prediction_flush_interval=int(
                    config.get("checkpoint", {}).get(
                        "flush_every", config.get("batch_size", 1)
                    )
                ),
            )
            shard_items = items[runner.shard_rank :: runner.shard_count]
            if not shard_items:
                raise ValueError(f"Evaluation shard {runner.shard_rank} has no samples.")
            result = runner.run(shard_items)
            result.artifacts.append(
                Artifact("dataset_audit", str(audit_path), "application/json")
            )
            return result
        finally:
            if pipeline is not None:
                pipeline.close()
