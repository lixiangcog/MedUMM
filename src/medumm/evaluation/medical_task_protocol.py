from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from medumm.medical.tasks import MedicalTaskSample, MedicalTaskType


MEDICAL_TASK_GROUPS = (
    "medical_task",
    "task_family",
    "specialty",
    "modality",
    "anatomy",
    "answer_type",
    "language",
)


@dataclass(frozen=True, slots=True)
class MedicalTaskProtocol:
    name: str = "medical_tasks"
    version: str = "1.0"
    metric_suite: str = "medical_task_core"
    metric_suite_version: str = "1.0"
    group_by: tuple[str, ...] = ("medical_task", "task_family", "specialty", "modality")
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    seed: int = 42
    calibration_bins: int = 10
    selective_thresholds: tuple[float, ...] = (0.5, 0.7, 0.9)
    minimum_group_samples: int = 1
    require_provenance: bool = False
    require_reference_provenance: bool = True
    require_deidentified: bool = False
    minimum_samples: int = 1
    minimum_samples_per_task: int = 1
    required_tasks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.version.strip():
            raise ValueError("Medical task protocol name and version cannot be empty.")
        if self.bootstrap_samples < 0:
            raise ValueError("bootstrap_samples cannot be negative.")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between zero and one.")
        if self.minimum_samples < 1 or self.minimum_samples_per_task < 1:
            raise ValueError("Medical task sample minimums must be at least one.")
        if self.calibration_bins < 2:
            raise ValueError("calibration_bins must be at least two.")
        if self.minimum_group_samples < 1:
            raise ValueError("minimum_group_samples must be at least one.")
        if any(not 0 <= value <= 1 for value in self.selective_thresholds):
            raise ValueError("selective_thresholds must be between zero and one.")
        unsupported_groups = sorted(set(self.group_by) - set(MEDICAL_TASK_GROUPS))
        if unsupported_groups:
            raise ValueError(f"Unsupported medical task group fields: {unsupported_groups}.")
        supported_tasks = {task.value for task in MedicalTaskType}
        unsupported_tasks = sorted(set(self.required_tasks) - supported_tasks)
        if unsupported_tasks:
            raise ValueError(f"Unsupported required medical tasks: {unsupported_tasks}.")

    @classmethod
    def from_config(
        cls,
        value: dict[str, Any] | None,
        *,
        seed: int,
        metric_suite_version: str,
    ) -> "MedicalTaskProtocol":
        config = dict(value or {})
        raw_groups = config.get(
            "group_by", ("medical_task", "task_family", "specialty", "modality")
        )
        raw_tasks = config.get("required_tasks", ())
        if not isinstance(raw_groups, (list, tuple)):
            raise ValueError("evaluation.protocol.group_by must be a list.")
        if not isinstance(raw_tasks, (list, tuple)):
            raise ValueError("evaluation.protocol.required_tasks must be a list.")
        raw_thresholds = config.get("selective_thresholds", (0.5, 0.7, 0.9))
        if not isinstance(raw_thresholds, (list, tuple)):
            raise ValueError("evaluation.protocol.selective_thresholds must be a list.")
        return cls(
            name=str(config.get("name", "medical_tasks")),
            version=str(config.get("version", "1.0")),
            metric_suite=str(config.get("metric_suite", "medical_task_core")),
            metric_suite_version=metric_suite_version,
            group_by=tuple(str(item) for item in raw_groups),
            bootstrap_samples=int(config.get("bootstrap_samples", 1000)),
            confidence_level=float(config.get("confidence_level", 0.95)),
            seed=int(config.get("seed", seed)),
            calibration_bins=int(config.get("calibration_bins", 10)),
            selective_thresholds=tuple(float(item) for item in raw_thresholds),
            minimum_group_samples=int(config.get("minimum_group_samples", 1)),
            require_provenance=bool(config.get("require_provenance", False)),
            require_reference_provenance=bool(
                config.get("require_reference_provenance", True)
            ),
            require_deidentified=bool(config.get("require_deidentified", False)),
            minimum_samples=int(config.get("minimum_samples", 1)),
            minimum_samples_per_task=int(config.get("minimum_samples_per_task", 1)),
            required_tasks=tuple(str(item) for item in raw_tasks),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["group_by"] = list(self.group_by)
        value["required_tasks"] = list(self.required_tasks)
        value["selective_thresholds"] = list(self.selective_thresholds)
        return value


def _resolved_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def audit_medical_task_dataset(
    samples: list[MedicalTaskSample],
    *,
    data_config: dict[str, Any],
    project_root: Path,
    dataset_fingerprint: str,
    protocol: MedicalTaskProtocol,
) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    manifest_path = _resolved_path(str(data_config.get("path", "")), project_root)
    raw_provenance_path = data_config.get("provenance")
    provenance_path = (
        _resolved_path(str(raw_provenance_path), project_root)
        if raw_provenance_path
        else manifest_path.parent / "provenance.json"
    )
    provenance: dict[str, Any] = {}
    if provenance_path.is_file():
        loaded = json.loads(provenance_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            provenance = loaded
        else:
            errors.append("Dataset provenance must be a JSON object.")
    elif protocol.require_provenance:
        errors.append("The medical task protocol requires a provenance document.")
    else:
        warnings.append("No dataset provenance document was discovered.")

    if protocol.require_deidentified and data_config.get("deidentified") is not True:
        errors.append("The medical task protocol requires deidentified: true.")
    elif data_config.get("deidentified") is not True:
        warnings.append("Dataset de-identification status was not explicitly affirmed.")
    if provenance.get("clinical_use") is True:
        errors.append("Clinical-use data cannot enter the research benchmark path.")
    if len(samples) < protocol.minimum_samples:
        errors.append(
            f"Dataset has {len(samples)} samples; protocol minimum is {protocol.minimum_samples}."
        )

    task_counts = Counter(sample.task.value for sample in samples)
    case_counts = Counter(sample.case_id for sample in samples if sample.case_id)
    for task in protocol.required_tasks:
        count = task_counts.get(task, 0)
        if count < protocol.minimum_samples_per_task:
            errors.append(
                f"Task {task} has {count} samples; required minimum is "
                f"{protocol.minimum_samples_per_task}."
            )
    missing_reference_provenance = [
        sample.sample_id for sample in samples if not sample.reference_provenance
    ]
    if missing_reference_provenance and protocol.require_reference_provenance:
        errors.append(
            f"{len(missing_reference_provenance)} samples lack reference provenance."
        )
    elif missing_reference_provenance:
        warnings.append(
            f"{len(missing_reference_provenance)} samples lack reference provenance."
        )
    heuristic_mappings = sum(
        sample.metadata.get("task_mapping", {}).get("method") == "heuristic"
        for sample in samples
        if isinstance(sample.metadata.get("task_mapping", {}), dict)
    )
    if heuristic_mappings:
        warnings.append(
            f"{heuristic_mappings} task labels are transparent heuristic mappings, not expert labels."
        )

    image_paths = [path for sample in samples for path in sample.image_paths]
    volume_paths = [path for sample in samples for path in sample.volume_paths]
    video_paths = [path for sample in samples for path in sample.video_paths]
    missing_images = sorted(path for path in image_paths if not Path(path).is_file())
    missing_volumes = sorted(path for path in volume_paths if not Path(path).is_file())
    missing_videos = sorted(path for path in video_paths if not Path(path).is_file())
    image_optional = {MedicalTaskType.PATIENT_COMMUNICATION}
    invalid_image_free = [
        sample.sample_id
        for sample in samples
        if not (sample.image_paths or sample.volume_paths or sample.video_paths)
        and sample.task not in image_optional
    ]
    if missing_images:
        errors.append(f"{len(missing_images)} referenced images are missing.")
    if missing_volumes:
        errors.append(f"{len(missing_volumes)} referenced volumes are missing.")
    if missing_videos:
        errors.append(f"{len(missing_videos)} referenced videos are missing.")
    if invalid_image_free:
        errors.append(f"{len(invalid_image_free)} image-dependent samples have no image.")

    diagnostic_without_evidence = [
        sample.sample_id
        for sample in samples
        if sample.task is MedicalTaskType.DIAGNOSTIC_REASONING
        and not sample.evidence
    ]
    if diagnostic_without_evidence:
        warnings.append(
            f"{len(diagnostic_without_evidence)} diagnostic samples have no evidence targets; "
            "diagnostic accuracy can be scored but reasoning task success cannot."
        )
    generation_without_concepts = [
        sample.sample_id
        for sample in samples
        if sample.task
        in {MedicalTaskType.REPORT_GENERATION, MedicalTaskType.PATIENT_COMMUNICATION}
        and not sample.concepts
    ]
    if generation_without_concepts:
        warnings.append(
            f"{len(generation_without_concepts)} long-form samples have no concept checklist."
        )

    status = "failed" if errors else "warning" if warnings else "passed"
    return {
        "schema_version": "1.0",
        "status": status,
        "dataset_fingerprint": dataset_fingerprint,
        "manifest": {
            "name": manifest_path.name,
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
        "sample_count": len(samples),
        "unique_id_count": len({sample.sample_id for sample in samples}),
        "case_count": len(case_counts),
        "samples_without_case_id": sum(sample.case_id is None for sample in samples),
        "multi_turn_case_count": sum(count > 1 for count in case_counts.values()),
        "maximum_turns_per_case": max(case_counts.values(), default=0),
        "reference_count": sum(len(sample.references) for sample in samples),
        "image_count": len(image_paths),
        "unique_image_count": len(set(image_paths)),
        "volume_count": len(volume_paths),
        "unique_volume_count": len(set(volume_paths)),
        "video_count": len(video_paths),
        "unique_video_count": len(set(video_paths)),
        "missing_images": missing_images,
        "missing_volumes": missing_volumes,
        "missing_videos": missing_videos,
        "image_dependent_samples_without_images": invalid_image_free,
        "task_distribution": dict(sorted(task_counts.items())),
        "task_family_distribution": dict(
            sorted(Counter(sample.task_family for sample in samples).items())
        ),
        "specialty_distribution": dict(
            sorted(Counter(sample.specialty for sample in samples).items())
        ),
        "reference_provenance": {
            "present": len(samples) - len(missing_reference_provenance),
            "missing_ids": missing_reference_provenance,
        },
        "structured_targets": {
            "concept_samples": sum(bool(sample.concepts) for sample in samples),
            "evidence_samples": sum(bool(sample.evidence) for sample in samples),
            "diagnostic_without_evidence": diagnostic_without_evidence,
            "long_form_without_concepts": generation_without_concepts,
        },
        "mapping": {
            "heuristic_count": heuristic_mappings,
            "expert_or_native_count": len(samples) - heuristic_mappings,
        },
        "governance": {
            "deidentified_declared": data_config.get("deidentified") is True,
            "research_only": provenance.get("clinical_use") is not True,
            "source": provenance.get("source"),
            "license": provenance.get("license"),
            "provenance_file": provenance_path.name if provenance_path.is_file() else None,
            "provenance_sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest()
            if provenance_path.is_file()
            else None,
            "resolved_revision": provenance.get("resolved_revision"),
        },
        "warnings": warnings,
        "errors": errors,
    }
