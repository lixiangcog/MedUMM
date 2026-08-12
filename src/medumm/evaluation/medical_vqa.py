from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from medumm.core.builtins import register_builtins
from medumm.core.config import execution_config
from medumm.core.contracts import EvaluationMode
from medumm.core.interfaces import BenchmarkAdapter
from medumm.core.registry import registry
from medumm.core.results import EvaluationResult, InferenceResult
from medumm.core.runtime import RuntimeContext
from medumm.evaluation.runner import EvaluationItem, EvaluationRunner
from medumm.inference import InferencePipeline
from medumm.medical import evaluate_answer, summarize_scores


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
        model_config = config.get("model")
        evaluation_config = config
        if not isinstance(data_config, dict) or not isinstance(model_config, dict):
            raise ValueError("Evaluation config requires data and model mappings.")
        dataset_name = str(data_config.get("adapter", "medical_vqa_jsonl"))
        dataset = registry.datasets.create(dataset_name)
        samples = dataset.load(data_config, runtime.project_root)
        dataset_fingerprint = dataset.fingerprint(data_config, runtime.project_root)
        backbone = str(model_config.get("backbone", "")).strip().lower()
        if not backbone:
            raise ValueError("Evaluation model requires a backbone.")
        mode = EvaluationMode(evaluation_config.get("mode", "full"))
        output_directory = Path(
            evaluation_config.get("output_directory", "outputs/evaluation/medical_vqa")
        )
        output_directory = (
            output_directory
            if output_directory.is_absolute()
            else runtime.project_root / output_directory
        )
        model_identity = {
            "backbone": backbone,
            "config": model_config.get("config", {}),
            "parameters": model_config.get("parameters", {}),
            "prompt_template": evaluation_config.get("prompt_template", "{question}"),
        }
        run_fingerprint = hashlib.sha256(
            json.dumps(
                {"dataset": dataset_fingerprint, "model": model_identity},
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
            runner = EvaluationRunner(
                benchmark=self.name,
                pipeline=pipeline,
                output_directory=output_directory,
                parser=_parse_output,
                scorer=lambda prediction, content: evaluate_answer(
                    prediction, content["references"], content["choices"]
                ),
                summarizer=summarize_scores,
                mode=mode,
                resume=bool(evaluation_config.get("resume", True)),
                fingerprint=run_fingerprint,
                metadata={
                    "dataset": dataset_name,
                    "dataset_fingerprint": dataset_fingerprint,
                    "model": backbone,
                    "clinical_use": False,
                },
                batch_size=int(evaluation_config.get("batch_size", 1)),
            )
            return runner.run(items)
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
