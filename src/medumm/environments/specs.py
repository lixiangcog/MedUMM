from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


_SHA = re.compile(r"^[0-9a-f]{40}$")
_PIN_OPERATORS = ("==", "===", " @ ")


class ValidationLevel(str, Enum):
    CONTRACT = "contract_validated"
    RESOLVED = "lock_resolved"
    CONTAINER = "container_built"
    IMPORT = "import_validated"
    RUNTIME = "runtime_validated"


@dataclass(frozen=True, slots=True)
class SourcePin:
    repository: str
    revision: str
    install: bool = True

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourcePin":
        repository = _https(value, "repository")
        revision = _text(value, "revision").lower()
        if not _SHA.fullmatch(revision):
            raise ValueError(
                f"Source revision for {repository!r} must be a 40-character commit SHA."
            )
        return cls(
            repository=repository,
            revision=revision,
            install=bool(value.get("install", True)),
        )


@dataclass(frozen=True, slots=True)
class EnvironmentSpec:
    model: str
    profile: str
    python: str
    platform: str
    accelerator: str
    cuda: str
    docker_base_image: str
    apptainer_base_image: str
    torch_index: str | None
    dependencies: tuple[str, ...]
    sources: tuple[SourcePin, ...]
    model_revision: str
    minimum_gpu_memory_gb: int
    recommended_gpus: int
    access: str
    imports: tuple[str, ...]
    validation: ValidationLevel
    evidence: str | None
    resolution: str
    notes: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EnvironmentSpec":
        dependencies = tuple(_nonempty_list(value, "dependencies"))
        for dependency in dependencies:
            _require_immutable_dependency(dependency)
        model_revision = _text(value, "model_revision").lower()
        if not _SHA.fullmatch(model_revision):
            raise ValueError(
                f"Model revision for {_text(value, 'model')!r} must be a 40-character SHA."
            )
        docker_base_image = _text(value, "docker_base_image")
        apptainer_base_image = _text(value, "apptainer_base_image")
        if "@sha256:" not in docker_base_image:
            raise ValueError(f"Docker base image must be digest-pinned: {docker_base_image!r}.")
        if "@sha256:" not in apptainer_base_image:
            raise ValueError(
                f"Apptainer base image must be digest-pinned: {apptainer_base_image!r}."
            )
        python = _text(value, "python")
        if not re.fullmatch(r"3\.\d+", python):
            raise ValueError(f"Python must pin a minor version, found {python!r}.")
        imports = tuple(_nonempty_list(value, "imports"))
        evidence = value.get("evidence")
        if evidence is not None and not str(evidence).strip():
            evidence = None
        return cls(
            model=_name(value, "model"),
            profile=_name(value, "profile"),
            python=python,
            platform=_text(value, "platform"),
            accelerator=_text(value, "accelerator"),
            cuda=_text(value, "cuda"),
            docker_base_image=docker_base_image,
            apptainer_base_image=apptainer_base_image,
            torch_index=(str(value["torch_index"]).strip() if value.get("torch_index") else None),
            dependencies=dependencies,
            sources=tuple(SourcePin.from_dict(item) for item in value.get("sources", [])),
            model_revision=model_revision,
            minimum_gpu_memory_gb=int(value.get("minimum_gpu_memory_gb", 0)),
            recommended_gpus=int(value.get("recommended_gpus", 1)),
            access=_text(value, "access"),
            imports=imports,
            validation=ValidationLevel(_text(value, "validation")),
            evidence=(str(evidence) if evidence is not None else None),
            resolution=_text(value, "resolution"),
            notes=tuple(str(item) for item in value.get("notes", [])),
        )

    def fingerprint(self) -> str:
        payload = "\n".join(
            (
                self.model,
                self.profile,
                self.python,
                self.platform,
                self.accelerator,
                self.cuda,
                self.docker_base_image,
                self.apptainer_base_image,
                self.torch_index or "",
                self.model_revision,
                *self.dependencies,
                *(f"{source.repository}@{source.revision}" for source in self.sources),
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def requirements(self) -> str:
        lines = [
            "# Generated from environments/models.yaml; do not edit by hand.",
            f"# model={self.model}",
            f"# profile={self.profile}",
            f"# contract_sha256={self.fingerprint()}",
        ]
        lines.extend(self.dependencies)
        for source in self.sources:
            if source.install:
                lines.append(f"git+{source.repository}.git@{source.revision}")
        return "\n".join(lines) + "\n"

    def source_manifest(self) -> str:
        lines = [
            "# Generated from environments/models.yaml; do not edit by hand.",
            f"# model={self.model}",
        ]
        for source in self.sources:
            lines.append(
                f"{source.repository}\t{source.revision}\tinstall={str(source.install).lower()}"
            )
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["validation"] = self.validation.value
        result["fingerprint"] = self.fingerprint()
        return result


class EnvironmentCatalog:
    def __init__(self, *, version: str, values: tuple[EnvironmentSpec, ...]) -> None:
        self.version = version
        self._values = values
        self._by_name = {item.model: item for item in values}
        if len(self._by_name) != len(values):
            raise ValueError("Duplicate model names in the environment catalog.")

    @classmethod
    def load(cls, path: str | Path) -> "EnvironmentCatalog":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("models"), list):
            raise ValueError("Environment catalog requires a top-level models list.")
        return cls(
            version=_text(raw, "catalog_version"),
            values=tuple(EnvironmentSpec.from_dict(item) for item in raw["models"]),
        )

    def get(self, name: str) -> EnvironmentSpec:
        normalized = name.strip().lower()
        try:
            return self._by_name[normalized]
        except KeyError as error:
            raise KeyError(f"Unknown model environment: {name!r}.") from error

    def names(self) -> list[str]:
        return sorted(self._by_name)

    def values(self) -> tuple[EnvironmentSpec, ...]:
        return tuple(self._by_name[name] for name in self.names())

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_version": self.version,
            "count": len(self._values),
            "models": [item.to_dict() for item in self.values()],
        }


def _text(value: dict[str, Any], key: str) -> str:
    result = str(value.get(key, "")).strip()
    if not result:
        raise ValueError(f"Environment field {key!r} cannot be empty.")
    return result


def _name(value: dict[str, Any], key: str) -> str:
    result = _text(value, key).lower()
    if not re.fullmatch(r"[a-z0-9_]+", result):
        raise ValueError(f"Environment field {key!r} is not a valid identifier: {result!r}.")
    return result


def _https(value: dict[str, Any], key: str) -> str:
    result = _text(value, key)
    if not result.startswith("https://"):
        raise ValueError(f"Environment field {key!r} must use HTTPS.")
    return result.removesuffix(".git")


def _nonempty_list(value: dict[str, Any], key: str) -> list[str]:
    result = value.get(key)
    if not isinstance(result, list) or not result:
        raise ValueError(f"Environment field {key!r} must be a non-empty list.")
    return [str(item).strip() for item in result]


def _require_immutable_dependency(dependency: str) -> None:
    lowered = dependency.casefold()
    if lowered.startswith(("-e ", "--editable ")):
        raise ValueError(f"Editable dependency is not reproducible: {dependency!r}.")
    if "git+" in lowered:
        revision = dependency.rsplit("@", 1)[-1].split("#", 1)[0]
        if not _SHA.fullmatch(revision):
            raise ValueError(f"Git dependency must use a full commit SHA: {dependency!r}.")
        return
    if lowered.startswith(("https://", "http://")):
        if "#sha256=" not in lowered:
            raise ValueError(f"URL dependency must include a SHA-256 hash: {dependency!r}.")
        return
    if not any(operator in dependency for operator in _PIN_OPERATORS):
        raise ValueError(f"Dependency must be exact-pinned: {dependency!r}.")
