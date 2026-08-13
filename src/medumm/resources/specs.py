from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from importlib.resources import files
from typing import Any, Generic, TypeVar

import yaml

from medumm.core.contracts import Modality, TaskType


class AccessLevel(str, Enum):
    OPEN = "open"
    GATED = "gated"
    CREDENTIALED = "credentialed"
    REQUEST = "request"


class IntegrationStatus(str, Enum):
    CATALOGED = "cataloged"
    INTERFACE_VALIDATED = "interface_validated"
    RUNTIME_VALIDATED = "runtime_validated"


class ModelRuntimeFamily(str, Enum):
    HF_IMAGE_TEXT = "hf_image_text_to_text"
    HF_CONTRASTIVE = "hf_contrastive"
    OPEN_CLIP = "open_clip"
    OFFICIAL_BRIDGE = "official_bridge"


class DatasetAdapterFamily(str, Enum):
    VQA = "vqa"
    MEDICAL_TASK = "medical_task"
    REPORT_GENERATION = "report_generation"
    CLASSIFICATION = "classification"
    RETRIEVAL = "retrieval"
    DETECTION_MEASUREMENT = "detection_measurement"
    VIDEO = "video"
    VOLUME = "volume"


@dataclass(frozen=True, slots=True)
class ModelResourceSpec:
    name: str
    display_name: str
    artifact_id: str
    source: str
    paper: str
    official_code: str | None
    license: str
    access: AccessLevel
    runtime_family: ModelRuntimeFamily
    status: IntegrationStatus
    tasks: tuple[TaskType, ...]
    input_modalities: tuple[Modality, ...]
    medical_domains: tuple[str, ...]
    parameters_b: float | None
    languages: tuple[str, ...]
    trust_remote_code: bool
    revision_policy: str
    notes: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelResourceSpec":
        return cls(
            name=_name(value),
            display_name=_required(value, "display_name"),
            artifact_id=_required(value, "artifact_id"),
            source=_url(value, "source"),
            paper=_url(value, "paper"),
            official_code=_optional_url(value, "official_code"),
            license=_required(value, "license"),
            access=AccessLevel(_required(value, "access")),
            runtime_family=ModelRuntimeFamily(_required(value, "runtime_family")),
            status=IntegrationStatus(_required(value, "status")),
            tasks=tuple(TaskType(item) for item in _nonempty_list(value, "tasks")),
            input_modalities=tuple(
                Modality(item) for item in _nonempty_list(value, "input_modalities")
            ),
            medical_domains=tuple(_nonempty_list(value, "medical_domains")),
            parameters_b=(
                float(value["parameters_b"]) if value.get("parameters_b") is not None else None
            ),
            languages=tuple(_nonempty_list(value, "languages")),
            trust_remote_code=bool(value.get("trust_remote_code", False)),
            revision_policy=str(value.get("revision_policy", "resolve_before_download")),
            notes=tuple(str(item) for item in value.get("notes", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["access"] = self.access.value
        result["runtime_family"] = self.runtime_family.value
        result["status"] = self.status.value
        result["tasks"] = [item.value for item in self.tasks]
        result["input_modalities"] = [item.value for item in self.input_modalities]
        result["medical_domains"] = list(self.medical_domains)
        result["languages"] = list(self.languages)
        result["notes"] = list(self.notes)
        return result


@dataclass(frozen=True, slots=True)
class DatasetResourceSpec:
    name: str
    display_name: str
    artifact_id: str
    source: str
    paper: str
    official_code: str | None
    license: str
    access: AccessLevel
    adapter_family: DatasetAdapterFamily
    status: IntegrationStatus
    benchmark: str
    tasks: tuple[str, ...]
    modalities: tuple[Modality, ...]
    medical_domains: tuple[str, ...]
    languages: tuple[str, ...]
    metrics: tuple[str, ...]
    revision_policy: str
    notes: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DatasetResourceSpec":
        return cls(
            name=_name(value),
            display_name=_required(value, "display_name"),
            artifact_id=_required(value, "artifact_id"),
            source=_url(value, "source"),
            paper=_url(value, "paper"),
            official_code=_optional_url(value, "official_code"),
            license=_required(value, "license"),
            access=AccessLevel(_required(value, "access")),
            adapter_family=DatasetAdapterFamily(_required(value, "adapter_family")),
            status=IntegrationStatus(_required(value, "status")),
            benchmark=_required(value, "benchmark"),
            tasks=tuple(_nonempty_list(value, "tasks")),
            modalities=tuple(Modality(item) for item in _nonempty_list(value, "modalities")),
            medical_domains=tuple(_nonempty_list(value, "medical_domains")),
            languages=tuple(_nonempty_list(value, "languages")),
            metrics=tuple(_nonempty_list(value, "metrics")),
            revision_policy=str(value.get("revision_policy", "resolve_before_download")),
            notes=tuple(str(item) for item in value.get("notes", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["access"] = self.access.value
        result["adapter_family"] = self.adapter_family.value
        result["status"] = self.status.value
        result["modalities"] = [item.value for item in self.modalities]
        for key in ("tasks", "medical_domains", "languages", "metrics", "notes"):
            result[key] = list(result[key])
        return result


Spec = TypeVar("Spec", ModelResourceSpec, DatasetResourceSpec)


class ResourceCatalog(Generic[Spec]):
    def __init__(self, *, kind: str, version: str, values: tuple[Spec, ...]) -> None:
        self.kind = kind
        self.version = version
        self._values = values
        self._by_name = {item.name: item for item in values}
        if len(self._by_name) != len(values):
            raise ValueError(f"Duplicate {kind} resource names in catalog.")

    def get(self, name: str) -> Spec:
        normalized = name.strip().lower()
        try:
            return self._by_name[normalized]
        except KeyError as error:
            raise KeyError(f"Unknown {self.kind} resource: {name!r}.") from error

    def names(self) -> list[str]:
        return sorted(self._by_name)

    def values(self) -> tuple[Spec, ...]:
        return tuple(self._by_name[name] for name in self.names())

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_version": self.version,
            "kind": self.kind,
            "count": len(self._values),
            "resources": [item.to_dict() for item in self.values()],
        }


def _required(value: dict[str, Any], key: str) -> str:
    result = str(value.get(key, "")).strip()
    if not result:
        raise ValueError(f"Catalog field {key!r} cannot be empty.")
    return result


def _name(value: dict[str, Any]) -> str:
    name = _required(value, "name").lower()
    if not all(character.islower() or character.isdigit() or character == "_" for character in name):
        raise ValueError(f"Invalid resource name: {name!r}.")
    return name


def _url(value: dict[str, Any], key: str) -> str:
    result = _required(value, key)
    if not result.startswith("https://"):
        raise ValueError(f"Catalog field {key!r} must use HTTPS: {result!r}.")
    return result


def _optional_url(value: dict[str, Any], key: str) -> str | None:
    raw = value.get(key)
    if raw is None or not str(raw).strip():
        return None
    return _url(value, key)


def _nonempty_list(value: dict[str, Any], key: str) -> list[str]:
    result = value.get(key)
    if not isinstance(result, list) or not result:
        raise ValueError(f"Catalog field {key!r} must be a non-empty list.")
    return [str(item) for item in result]


def _read_catalog(filename: str) -> dict[str, Any]:
    resource = files("medumm.resources").joinpath("catalog", filename)
    loaded = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in resource catalog {filename}.")
    return loaded


def _load_models() -> ResourceCatalog[ModelResourceSpec]:
    raw = _read_catalog("models.yaml")
    values = raw.get("models")
    if not isinstance(values, list):
        raise ValueError("Model catalog requires a models list.")
    return ResourceCatalog(
        kind="model",
        version=_required(raw, "catalog_version"),
        values=tuple(ModelResourceSpec.from_dict(item) for item in values),
    )


def _load_datasets() -> ResourceCatalog[DatasetResourceSpec]:
    raw = _read_catalog("datasets.yaml")
    values = raw.get("datasets")
    if not isinstance(values, list):
        raise ValueError("Dataset catalog requires a datasets list.")
    return ResourceCatalog(
        kind="dataset",
        version=_required(raw, "catalog_version"),
        values=tuple(DatasetResourceSpec.from_dict(item) for item in values),
    )


MODEL_RESOURCES = _load_models()
DATASET_RESOURCES = _load_datasets()


def resource_catalog(kind: str = "all") -> dict[str, Any]:
    normalized = kind.strip().lower()
    if normalized == "model":
        return MODEL_RESOURCES.to_dict()
    if normalized == "dataset":
        return DATASET_RESOURCES.to_dict()
    if normalized != "all":
        raise ValueError("Resource kind must be model, dataset, or all.")
    return {
        "catalog_version": MODEL_RESOURCES.version,
        "counts": {
            "models": len(MODEL_RESOURCES.values()),
            "datasets": len(DATASET_RESOURCES.values()),
        },
        "models": [item.to_dict() for item in MODEL_RESOURCES.values()],
        "datasets": [item.to_dict() for item in DATASET_RESOURCES.values()],
    }
