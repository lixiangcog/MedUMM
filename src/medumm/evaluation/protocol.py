from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from medumm.medical.data import MedicalVQASample


DEFAULT_GROUPS = ("modality", "category", "answer_type", "language")


@dataclass(frozen=True, slots=True)
class EvaluationProtocol:
    """Resolved, serializable rules that make a medical score reproducible."""

    name: str = "medical_vqa"
    version: str = "1.0"
    metric_suite: str = "medical_vqa_core"
    metric_suite_version: str = "1.0"
    group_by: tuple[str, ...] = DEFAULT_GROUPS
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    seed: int = 42
    require_provenance: bool = False
    require_deidentified: bool = False
    minimum_samples: int = 1

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.version.strip():
            raise ValueError("Evaluation protocol name and version cannot be empty.")
        if self.bootstrap_samples < 0:
            raise ValueError("bootstrap_samples cannot be negative.")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between zero and one.")
        if self.minimum_samples < 1:
            raise ValueError("minimum_samples must be at least one.")
        unsupported = sorted(set(self.group_by) - set(DEFAULT_GROUPS))
        if unsupported:
            raise ValueError(f"Unsupported medical VQA group fields: {unsupported}.")

    @classmethod
    def from_config(
        cls,
        value: dict[str, Any] | None,
        *,
        seed: int,
        metric_suite_version: str,
    ) -> "EvaluationProtocol":
        config = dict(value or {})
        raw_groups = config.get("group_by", DEFAULT_GROUPS)
        if not isinstance(raw_groups, (list, tuple)):
            raise ValueError("evaluation.protocol.group_by must be a list.")
        return cls(
            name=str(config.get("name", "medical_vqa")),
            version=str(config.get("version", "1.0")),
            metric_suite=str(config.get("metric_suite", "medical_vqa_core")),
            metric_suite_version=metric_suite_version,
            group_by=tuple(str(item) for item in raw_groups),
            bootstrap_samples=int(config.get("bootstrap_samples", 1000)),
            confidence_level=float(config.get("confidence_level", 0.95)),
            seed=int(config.get("seed", seed)),
            require_provenance=bool(config.get("require_provenance", False)),
            require_deidentified=bool(config.get("require_deidentified", False)),
            minimum_samples=int(config.get("minimum_samples", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["group_by"] = list(self.group_by)
        return value


def _resolved_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def _provenance_path(data_config: dict[str, Any], project_root: Path) -> Path | None:
    explicit = data_config.get("provenance")
    if explicit:
        return _resolved_path(str(explicit), project_root)
    source = _resolved_path(str(data_config.get("path", "")), project_root)
    candidate = source.parent / "provenance.json"
    return candidate if candidate.is_file() else None


def _distribution(samples: list[MedicalVQASample], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(getattr(sample, field, "unknown")) for sample in samples).items()))


def audit_medical_vqa_dataset(
    samples: list[MedicalVQASample],
    *,
    data_config: dict[str, Any],
    project_root: Path,
    dataset_fingerprint: str,
    protocol: EvaluationProtocol,
) -> dict[str, Any]:
    """Build a machine-readable data-quality and governance gate."""

    warnings: list[str] = []
    errors: list[str] = []
    provenance_path = _provenance_path(data_config, project_root)
    provenance: dict[str, Any] = {}
    if provenance_path is not None and provenance_path.is_file():
        loaded = json.loads(provenance_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            errors.append("Dataset provenance must be a JSON object.")
        else:
            provenance = loaded
    elif protocol.require_provenance:
        errors.append("The evaluation protocol requires a provenance document.")
    else:
        warnings.append("No dataset provenance document was discovered.")

    if protocol.require_deidentified and data_config.get("deidentified") is not True:
        errors.append("The evaluation protocol requires deidentified: true.")
    elif data_config.get("deidentified") is not True:
        warnings.append("Dataset de-identification status was not explicitly affirmed.")
    if provenance.get("clinical_use") is True:
        errors.append("A dataset marked for clinical use cannot enter the research benchmark path.")
    if len(samples) < protocol.minimum_samples:
        errors.append(
            f"Dataset has {len(samples)} samples; protocol minimum is {protocol.minimum_samples}."
        )

    image_paths = [path for sample in samples for path in sample.image_paths]
    missing_images = sorted(path for path in image_paths if not Path(path).is_file())
    samples_without_images = [sample.sample_id for sample in samples if not sample.image_paths]
    if missing_images:
        errors.append(f"{len(missing_images)} referenced images are missing.")
    if samples_without_images:
        errors.append(f"{len(samples_without_images)} medical VQA samples have no image.")
    unknown_fields = {
        field: sum(str(getattr(sample, field, "unknown")).strip().casefold() in {"", "unknown"} for sample in samples)
        for field in DEFAULT_GROUPS
    }
    for field, count in unknown_fields.items():
        if count:
            warnings.append(f"{count} samples have unknown {field} metadata.")

    manifest_path = _resolved_path(str(data_config.get("path", "")), project_root)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    answer_counts = Counter(
        "closed" if sample.choices else "open" for sample in samples
    )
    status = "failed" if errors else "warning" if warnings else "passed"
    return {
        "schema_version": "1.0",
        "status": status,
        "dataset_fingerprint": dataset_fingerprint,
        "manifest": {
            "name": manifest_path.name,
            "sha256": manifest_sha256,
        },
        "sample_count": len(samples),
        "unique_id_count": len({sample.sample_id for sample in samples}),
        "reference_count": sum(len(sample.answers) for sample in samples),
        "image_count": len(image_paths),
        "unique_image_count": len(set(image_paths)),
        "samples_without_images": samples_without_images,
        "missing_images": missing_images,
        "question_answer_format": dict(sorted(answer_counts.items())),
        "distributions": {
            field: _distribution(samples, field) for field in DEFAULT_GROUPS
        },
        "unknown_metadata": unknown_fields,
        "governance": {
            "deidentified_declared": data_config.get("deidentified") is True,
            "research_only": provenance.get("clinical_use") is not True,
            "source": provenance.get("source"),
            "license": provenance.get("license"),
            "provenance_file": provenance_path.name if provenance_path else None,
            "provenance_sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest()
            if provenance_path and provenance_path.is_file()
            else None,
            "resolved_revision": provenance.get("resolved_revision"),
        },
        "warnings": warnings,
        "errors": errors,
    }
