from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from medumm.medical.tasks import MedicalTaskType


class AlignmentObjective(str, Enum):
    SFT = "sft"
    DPO = "dpo"
    SIMPO = "simpo"
    ORPO = "orpo"
    CLINICAL_DPO = "clinical_dpo"

    @property
    def requires_rejected(self) -> bool:
        return self is not self.SFT

    @property
    def uses_reference_policy(self) -> bool:
        return self in {self.DPO, self.CLINICAL_DPO}


@dataclass(slots=True)
class MedicalAlignmentSample:
    sample_id: str
    prompt: str
    chosen: str
    rejected: str | None = None
    medical_task: MedicalTaskType | None = None
    specialty: str = "unknown"
    safety_categories: list[str] = field(default_factory=list)
    preference_rationale: str | None = None
    label_source: str = "unknown"
    clinical_relevance: float = 1.0
    image_paths: list[str] = field(default_factory=list)
    modality: str = "text"
    anatomy: str = "unknown"
    source_name: str = "default"
    source_weight: float = 1.0
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AlignmentDataBundle:
    samples: list[MedicalAlignmentSample]
    fingerprint: str
    sources: list[dict[str, Any]]
    audit: dict[str, Any]


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


def _message_text(value: Any, *, role: str | None = None) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, list):
        return ""
    candidates = []
    for message in value:
        if not isinstance(message, dict):
            continue
        message_role = str(message.get("role", "")).casefold()
        if role is None or message_role == role.casefold():
            content = str(message.get("content", "")).strip()
            if content:
                candidates.append(content)
    return candidates[-1] if candidates else ""


def _string_list(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return sorted(
        {str(item).strip() for item in values if item is not None and str(item).strip()}
    )


def _resolved(path: str | Path, project_root: Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else project_root / value


def _load_source(
    source: dict[str, Any],
    *,
    project_root: Path,
    default_deidentified: bool,
) -> tuple[list[MedicalAlignmentSample], dict[str, Any]]:
    name = str(source.get("name") or Path(str(source.get("path", ""))).stem).strip()
    if not name:
        raise ValueError("Every alignment data source requires a name or path.")
    path = _resolved(str(source.get("path", "")), project_root)
    if not path.is_file():
        raise FileNotFoundError(f"Alignment source not found: {path}")
    weight = float(source.get("weight", 1.0))
    if weight <= 0:
        raise ValueError(f"Alignment source {name} weight must be positive.")
    limit = int(source.get("max_samples", 0) or 0)
    values = _records(path)
    if limit:
        values = values[:limit]
    samples: list[MedicalAlignmentSample] = []
    image_root = _resolved(source.get("image_root", path.parent), project_root)
    validate_images = bool(source.get("validate_images", True))
    for index, record in enumerate(values):
        source_sample_id = str(
            record.get("id", record.get("prompt_id", f"row-{index}"))
        ).strip()
        sample_id = f"{name}:{source_sample_id}"
        prompt = _message_text(record.get("prompt"))
        if not prompt:
            prompt = _message_text(record.get("chosen"), role="user")
        chosen = _message_text(record.get("chosen"), role="assistant")
        rejected = _message_text(record.get("rejected"), role="assistant") or None
        if not sample_id or not prompt or not chosen:
            raise ValueError(f"Alignment sample {name}:{index} lacks id, prompt, or chosen text.")
        if rejected is not None and chosen.strip() == rejected.strip():
            raise ValueError(f"Alignment sample {sample_id} has identical preference responses.")
        raw_task = record.get("medical_task")
        task = MedicalTaskType(str(raw_task)) if raw_task else None
        raw_provenance = record.get("preference_provenance", record.get("provenance", {}))
        raw_metadata = record.get("metadata", {})
        raw_images = record.get("images", record.get("image", []))
        raw_images = raw_images if isinstance(raw_images, list) else [raw_images]
        image_paths: list[str] = []
        for value in raw_images:
            if value in {None, ""}:
                continue
            image_path = Path(str(value)).expanduser()
            image_path = image_path if image_path.is_absolute() else image_root / image_path
            if validate_images and not image_path.is_file():
                raise FileNotFoundError(
                    f"Image for alignment sample {sample_id} not found: {image_path}"
                )
            image_paths.append(str(image_path))
        samples.append(
            MedicalAlignmentSample(
                sample_id=sample_id,
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                medical_task=task,
                specialty=str(record.get("specialty", "unknown")),
                safety_categories=_string_list(record.get("safety_categories", [])),
                preference_rationale=str(
                    record.get("preference_rationale", record.get("feedback", ""))
                ).strip()
                or None,
                label_source=str(record.get("label_source", "unknown")).casefold(),
                clinical_relevance=float(record.get("clinical_relevance", 1.0)),
                image_paths=image_paths,
                modality=str(record.get("modality", "image" if image_paths else "text")),
                anatomy=str(record.get("anatomy", "unknown")),
                source_name=name,
                source_weight=weight,
                provenance=dict(raw_provenance) if isinstance(raw_provenance, dict) else {},
                metadata=dict(raw_metadata) if isinstance(raw_metadata, dict) else {},
            )
        )
    provenance_path = source.get("provenance")
    if provenance_path:
        provenance_file = _resolved(str(provenance_path), project_root)
    else:
        provenance_file = path.parent / "provenance.json"
    provenance: dict[str, Any] = {}
    if provenance_file.is_file():
        loaded = json.loads(provenance_file.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Alignment provenance must be an object: {provenance_file}")
        provenance = loaded
    return samples, {
        "name": name,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sample_count": len(samples),
        "weight": weight,
        "deidentified": bool(source.get("deidentified", default_deidentified)),
        "license": source.get("license", provenance.get("license")),
        "dataset_id": source.get("dataset_id", provenance.get("dataset")),
        "revision": source.get("revision", provenance.get("resolved_revision")),
        "provenance_file": str(provenance_file) if provenance_file.is_file() else None,
        "provenance": provenance,
    }


def load_alignment_data(
    config: dict[str, Any],
    *,
    project_root: Path,
    objective: AlignmentObjective,
) -> AlignmentDataBundle:
    raw_sources = config.get("mixtures")
    if raw_sources is None:
        raw_sources = [config]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("Alignment data.mixtures must be a non-empty list.")
    if not all(isinstance(item, dict) for item in raw_sources):
        raise ValueError("Every alignment mixture source must be a mapping.")
    samples: list[MedicalAlignmentSample] = []
    source_details: list[dict[str, Any]] = []
    source_names = [
        str(source.get("name") or Path(str(source.get("path", ""))).stem).strip()
        for source in raw_sources
    ]
    duplicate_source_names = sorted(
        name for name, count in Counter(source_names).items() if count > 1
    )
    if duplicate_source_names:
        raise ValueError(f"Duplicate alignment source names: {duplicate_source_names}")
    for source in raw_sources:
        loaded, details = _load_source(
            source,
            project_root=project_root,
            default_deidentified=bool(config.get("deidentified", False)),
        )
        samples.extend(loaded)
        source_details.append(details)
    identifiers = [sample.sample_id for sample in samples]
    duplicate_ids = sorted(
        identifier for identifier, count in Counter(identifiers).items() if count > 1
    )
    missing_rejected = [sample.sample_id for sample in samples if not sample.rejected]
    if objective.requires_rejected and missing_rejected:
        raise ValueError(
            f"{objective.value} requires rejected responses; missing for "
            f"{len(missing_rejected)} samples."
        )
    if not samples:
        raise ValueError("Alignment dataset contains no samples.")
    if duplicate_ids:
        raise ValueError(f"Duplicate alignment sample ids: {duplicate_ids[:5]}")
    invalid_relevance = [
        sample.sample_id
        for sample in samples
        if not math.isfinite(sample.clinical_relevance)
        or sample.clinical_relevance <= 0
    ]
    missing_preference_provenance = [
        sample.sample_id for sample in samples if not sample.provenance
    ]
    missing_preference_rationale = [
        sample.sample_id for sample in samples if not sample.preference_rationale
    ]
    missing_source_provenance = [
        source["name"] for source in source_details if not source["provenance_file"]
    ]
    undeidentified_sources = [
        source["name"] for source in source_details if not source["deidentified"]
    ]
    missing_licenses = [source["name"] for source in source_details if not source["license"]]
    errors: list[str] = []
    warnings: list[str] = []
    require_provenance = bool(config.get("require_provenance", True))
    require_preference_provenance = bool(
        config.get("require_preference_provenance", objective.requires_rejected)
    )
    require_deidentified = bool(config.get("require_deidentified", True))
    require_preference_rationale = bool(
        config.get("require_preference_rationale", objective.requires_rejected)
    )
    if invalid_relevance:
        errors.append(f"{len(invalid_relevance)} samples have non-positive relevance weights.")
    if missing_source_provenance and require_provenance:
        errors.append(f"Sources lack provenance documents: {missing_source_provenance}.")
    elif missing_source_provenance:
        warnings.append(f"Sources lack provenance documents: {missing_source_provenance}.")
    if missing_preference_provenance and require_preference_provenance:
        errors.append(
            f"{len(missing_preference_provenance)} samples lack preference provenance."
        )
    elif missing_preference_provenance:
        warnings.append(
            f"{len(missing_preference_provenance)} samples lack preference provenance."
        )
    if missing_preference_rationale and require_preference_rationale:
        errors.append(
            f"{len(missing_preference_rationale)} samples lack preference rationales."
        )
    elif missing_preference_rationale:
        warnings.append(
            f"{len(missing_preference_rationale)} samples lack preference rationales."
        )
    if undeidentified_sources and require_deidentified:
        errors.append(f"Sources are not declared deidentified: {undeidentified_sources}.")
    if missing_licenses:
        errors.append(f"Sources lack license declarations: {missing_licenses}.")
    non_expert = sum(sample.label_source not in {"expert", "clinician"} for sample in samples)
    if non_expert:
        warnings.append(
            f"{non_expert} preference labels are not declared clinician/expert annotations."
        )
    if objective is AlignmentObjective.CLINICAL_DPO and not any(
        sample.clinical_relevance != 1.0 for sample in samples
    ):
        warnings.append("clinical_dpo received only unit relevance weights.")
    digest = hashlib.sha256()
    digest.update(objective.value.encode())
    for source in source_details:
        stable_source = {
            key: value
            for key, value in source.items()
            if key not in {"path", "provenance_file"}
        }
        digest.update(json.dumps(stable_source, sort_keys=True, default=str).encode())
    audit = {
        "schema_version": "1.0",
        "status": "failed" if errors else "warning" if warnings else "passed",
        "objective": objective.value,
        "sample_count": len(samples),
        "preference_pair_count": len(samples) - len(missing_rejected),
        "source_count": len(source_details),
        "sources": source_details,
        "label_source_distribution": dict(
            sorted(Counter(sample.label_source for sample in samples).items())
        ),
        "medical_task_distribution": dict(
            sorted(
                Counter(
                    sample.medical_task.value if sample.medical_task else "unknown"
                    for sample in samples
                ).items()
            )
        ),
        "specialty_distribution": dict(
            sorted(Counter(sample.specialty for sample in samples).items())
        ),
        "modality_distribution": dict(
            sorted(Counter(sample.modality for sample in samples).items())
        ),
        "image_sample_count": sum(bool(sample.image_paths) for sample in samples),
        "safety_category_distribution": dict(
            sorted(
                Counter(
                    category
                    for sample in samples
                    for category in sample.safety_categories
                ).items()
            )
        ),
        "preference_rationale_count": sum(
            sample.preference_rationale is not None for sample in samples
        ),
        "preference_provenance_count": len(samples)
        - len(missing_preference_provenance),
        "clinician_or_expert_count": len(samples) - non_expert,
        "clinical_relevance": {
            "minimum": min(sample.clinical_relevance for sample in samples),
            "maximum": max(sample.clinical_relevance for sample in samples),
            "mean": sum(sample.clinical_relevance for sample in samples) / len(samples),
        },
        "warnings": warnings,
        "errors": errors,
    }
    audit["dataset_fingerprint"] = digest.hexdigest()
    return AlignmentDataBundle(samples, digest.hexdigest(), source_details, audit)


def deterministic_epoch_samples(
    samples: list[MedicalAlignmentSample],
    *,
    seed: int,
    epoch: int,
    epoch_size: int | None = None,
) -> list[MedicalAlignmentSample]:
    """Sample a weighted mixture deterministically while cycling within sources."""

    if not samples:
        return []
    groups: dict[str, list[MedicalAlignmentSample]] = {}
    source_weights: dict[str, float] = {}
    for sample in samples:
        groups.setdefault(sample.source_name, []).append(sample)
        source_weights[sample.source_name] = sample.source_weight
    generator = random.Random(seed + epoch)
    pools: dict[str, list[MedicalAlignmentSample]] = {}
    positions = {name: 0 for name in groups}
    for name, values in groups.items():
        pools[name] = list(values)
        generator.shuffle(pools[name])
    names = sorted(groups)
    weights = [source_weights[name] for name in names]
    size = int(epoch_size or len(samples))
    result: list[MedicalAlignmentSample] = []
    for name in generator.choices(names, weights=weights, k=size):
        if positions[name] >= len(pools[name]):
            generator.shuffle(pools[name])
            positions[name] = 0
        result.append(pools[name][positions[name]])
        positions[name] += 1
    return result


def alignment_sample_to_dict(sample: MedicalAlignmentSample) -> dict[str, Any]:
    value = asdict(sample)
    value["medical_task"] = sample.medical_task.value if sample.medical_task else None
    return value
