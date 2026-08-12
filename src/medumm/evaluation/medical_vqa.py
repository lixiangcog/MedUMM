from __future__ import annotations

from pathlib import Path
from typing import Any

from medumm.core.config import find_project_root
from medumm.evaluation.runner import EvaluationItem, EvaluationRunner
from medumm.inference import InferencePipeline
from medumm.medical import evaluate_answer, load_medical_vqa, summarize_scores


def _parse_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        understandings = output.get("understandings")
        if isinstance(understandings, list) and understandings:
            return str(understandings[0].get("response", ""))
    return str(output)


def run_medical_vqa(
    config: dict[str, Any],
    *,
    config_path: str | Path,
) -> dict[str, Any]:
    root = find_project_root(config_path)
    data_config = config.get("data")
    model_config = config.get("model")
    evaluation_config = config.get("evaluation", {})
    if not isinstance(data_config, dict) or not isinstance(model_config, dict):
        raise ValueError("Evaluation config requires data and model mappings.")
    samples = load_medical_vqa(data_config, project_root=root)
    backbone = str(model_config.get("backbone", ""))
    if not backbone:
        raise ValueError("Evaluation model requires a backbone.")
    output_directory = Path(evaluation_config.get("output_directory", "outputs/evaluation/medical_vqa"))
    output_directory = output_directory if output_directory.is_absolute() else root / output_directory
    items = [
        EvaluationItem(
            sample_id=sample.sample_id,
            request={
                "task": "understanding",
                "prompt": sample.question,
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
    with InferencePipeline(backbone, dict(model_config.get("config", {}))) as pipeline:
        runner = EvaluationRunner(
            benchmark="medical_vqa",
            pipeline=pipeline,
            output_directory=output_directory,
            parser=_parse_output,
            scorer=lambda prediction, content: evaluate_answer(
                prediction, content["references"], content["choices"]
            ),
            summarizer=summarize_scores,
            resume=bool(evaluation_config.get("resume", True)),
        )
        return runner.run(items)
