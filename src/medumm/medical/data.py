from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class MedicalVQASample:
    sample_id: str
    question: str
    image_paths: list[str]
    answers: list[str]
    choices: dict[str, str] = field(default_factory=dict)
    answer_type: str = "open"
    modality: str = "unknown"
    category: str = "unknown"
    language: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


def _records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    else:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        values = loaded.get("data") if isinstance(loaded, dict) else loaded
    if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
        raise ValueError(f"Expected a list of objects in {path}.")
    return values


def _choices(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {str(key).upper(): str(option) for key, option in value.items()}
    if isinstance(value, list):
        return {chr(65 + index): str(option) for index, option in enumerate(value)}
    return {}


def load_medical_vqa(
    config: dict[str, Any],
    *,
    project_root: Path,
) -> list[MedicalVQASample]:
    """Load local JSON/JSONL records into the MedUMM medical VQA schema."""

    if str(config.get("source", "jsonl")) not in {"json", "jsonl"}:
        raise ValueError("The medical_vqa_jsonl adapter supports local JSON and JSONL only.")
    data_path = Path(str(config.get("path", ""))).expanduser()
    data_path = data_path if data_path.is_absolute() else project_root / data_path
    if not data_path.is_file():
        raise FileNotFoundError(f"Medical VQA dataset not found: {data_path}")
    image_root = Path(str(config.get("image_root", data_path.parent))).expanduser()
    image_root = image_root if image_root.is_absolute() else project_root / image_root
    limit = int(config.get("max_samples", 0) or 0)

    samples: list[MedicalVQASample] = []
    identifiers: set[str] = set()
    for index, record in enumerate(_records(data_path)):
        sample_id = str(record.get("id", index))
        if sample_id in identifiers:
            raise ValueError(f"Duplicate sample id: {sample_id}")
        identifiers.add(sample_id)
        question = str(record.get("question", "")).strip()
        if not question:
            raise ValueError(f"Sample {sample_id} has no question.")
        raw_answers = record.get("answers", record.get("answer"))
        answers = raw_answers if isinstance(raw_answers, list) else [raw_answers]
        answers = [str(answer).strip() for answer in answers if answer is not None]
        if not answers:
            raise ValueError(f"Sample {sample_id} has no reference answer.")
        raw_images = record.get("images", record.get("image", []))
        raw_images = raw_images if isinstance(raw_images, list) else [raw_images]
        paths = []
        for value in raw_images:
            if value in {None, ""}:
                continue
            path = Path(str(value)).expanduser()
            path = path if path.is_absolute() else image_root / path
            if bool(config.get("validate_images", True)) and not path.is_file():
                raise FileNotFoundError(f"Image for sample {sample_id} not found: {path}")
            paths.append(str(path))
        samples.append(
            MedicalVQASample(
                sample_id=sample_id,
                question=question,
                image_paths=paths,
                answers=answers,
                choices=_choices(record.get("choices")),
                answer_type=str(record.get("answer_type", "open")),
                modality=str(record.get("modality", "unknown")),
                category=str(record.get("category", "unknown")),
                language=str(record.get("language", "unknown")),
                metadata=dict(record.get("metadata", {}))
                if isinstance(record.get("metadata", {}), dict)
                else {},
            )
        )
        if limit and len(samples) >= limit:
            break
    if not samples:
        raise ValueError("Medical VQA dataset contains no samples.")
    return samples
