from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class MedicalTaskType(str, Enum):
    """Clinical intent carried by an understanding request.

    These tasks describe the expected medical output, not the backbone
    architecture.  In particular, they are deliberately not natural-image
    class labels.
    """

    FINDING_ASSESSMENT = "finding_assessment"
    CLINICAL_DESCRIPTION = "clinical_description"
    ANATOMY_LOCALIZATION = "anatomy_localization"
    QUANTITATIVE_ASSESSMENT = "quantitative_assessment"
    IMAGE_CONTEXT = "image_context"
    DIAGNOSTIC_REASONING = "diagnostic_reasoning"
    REPORT_GENERATION = "report_generation"
    PATIENT_COMMUNICATION = "patient_communication"

    @property
    def family(self) -> str:
        if self in {
            self.FINDING_ASSESSMENT,
            self.CLINICAL_DESCRIPTION,
            self.ANATOMY_LOCALIZATION,
            self.QUANTITATIVE_ASSESSMENT,
            self.IMAGE_CONTEXT,
        }:
            return "perception"
        if self is self.DIAGNOSTIC_REASONING:
            return "reasoning"
        return "generation"


@dataclass(slots=True)
class MedicalTaskSample:
    sample_id: str
    task: MedicalTaskType
    prompt: str
    image_paths: list[str]
    references: list[str]
    specialty: str = "unknown"
    modality: str = "unknown"
    anatomy: str = "unknown"
    concepts: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    choices: dict[str, str] = field(default_factory=dict)
    answer_type: str = "open"
    language: str = "unknown"
    case_id: str | None = None
    turn_index: int | None = None
    reference_provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def task_family(self) -> str:
        return self.task.family


def _records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        values = loaded.get("data") if isinstance(loaded, dict) else loaded
    if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
        raise ValueError(f"Expected a list of objects in {path}.")
    return values


def _string_list(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in values if item is not None and str(item).strip()]


def _choices(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {str(key).upper(): str(option) for key, option in value.items()}
    if isinstance(value, list):
        return {chr(65 + index): str(option) for index, option in enumerate(value)}
    return {}


def load_medical_tasks(
    config: dict[str, Any],
    *,
    project_root: Path,
) -> list[MedicalTaskSample]:
    """Load the v0.6 task-aware local JSON/JSONL contract."""

    if str(config.get("source", "jsonl")) not in {"json", "jsonl"}:
        raise ValueError("The medical_tasks_jsonl adapter supports local JSON and JSONL only.")
    data_path = Path(str(config.get("path", ""))).expanduser()
    data_path = data_path if data_path.is_absolute() else project_root / data_path
    if not data_path.is_file():
        raise FileNotFoundError(f"Medical task dataset not found: {data_path}")
    image_root = Path(str(config.get("image_root", data_path.parent))).expanduser()
    image_root = image_root if image_root.is_absolute() else project_root / image_root
    limit = int(config.get("max_samples", 0) or 0)

    samples: list[MedicalTaskSample] = []
    identifiers: set[str] = set()
    for index, record in enumerate(_records(data_path)):
        sample_id = str(record.get("id", index))
        if sample_id in identifiers:
            raise ValueError(f"Duplicate sample id: {sample_id}")
        identifiers.add(sample_id)
        try:
            task = MedicalTaskType(str(record.get("task", "")))
        except ValueError as error:
            supported = ", ".join(item.value for item in MedicalTaskType)
            raise ValueError(
                f"Sample {sample_id} has unsupported medical task; expected one of: {supported}."
            ) from error
        prompt = str(record.get("prompt", record.get("question", ""))).strip()
        if not prompt:
            raise ValueError(f"Sample {sample_id} has no prompt.")
        references = _string_list(
            record.get("references", record.get("answers", record.get("answer")))
        )
        if not references:
            raise ValueError(f"Sample {sample_id} has no reference output.")
        raw_images = record.get("images", record.get("image", []))
        raw_images = raw_images if isinstance(raw_images, list) else [raw_images]
        paths: list[str] = []
        for value in raw_images:
            if value in {None, ""}:
                continue
            path = Path(str(value)).expanduser()
            path = path if path.is_absolute() else image_root / path
            if bool(config.get("validate_images", True)) and not path.is_file():
                raise FileNotFoundError(f"Image for sample {sample_id} not found: {path}")
            paths.append(str(path))
        raw_provenance = record.get("reference_provenance", {})
        raw_metadata = record.get("metadata", {})
        samples.append(
            MedicalTaskSample(
                sample_id=sample_id,
                task=task,
                prompt=prompt,
                image_paths=paths,
                references=references,
                specialty=str(record.get("specialty", "unknown")),
                modality=str(record.get("modality", "unknown")),
                anatomy=str(record.get("anatomy", "unknown")),
                concepts=_string_list(record.get("concepts", [])),
                evidence=_string_list(record.get("evidence", [])),
                choices=_choices(record.get("choices")),
                answer_type=str(record.get("answer_type", "open")),
                language=str(record.get("language", "unknown")),
                case_id=str(record["case_id"]) if record.get("case_id") is not None else None,
                turn_index=int(record["turn_index"])
                if record.get("turn_index") is not None
                else None,
                reference_provenance=dict(raw_provenance)
                if isinstance(raw_provenance, dict)
                else {},
                metadata=dict(raw_metadata) if isinstance(raw_metadata, dict) else {},
            )
        )
        if limit and len(samples) >= limit:
            break
    if not samples:
        raise ValueError("Medical task dataset contains no samples.")
    return samples
