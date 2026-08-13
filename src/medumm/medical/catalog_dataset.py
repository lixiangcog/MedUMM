from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from medumm.core.interfaces import DatasetAdapter
from medumm.medical.dataset import MedicalTasksDatasetAdapter, MedicalVQADatasetAdapter
from medumm.resources import AccessLevel, DATASET_RESOURCES, DatasetAdapterFamily


class CatalogDatasetAdapter(DatasetAdapter):
    """Dataset-specific registration over one normalized local manifest contract."""

    def __init__(self, resource_name: str) -> None:
        self.spec = DATASET_RESOURCES.get(resource_name)
        self.name = self.spec.name

    def _validate(self, config: dict[str, Any]) -> None:
        if self.spec.access is not AccessLevel.OPEN and not bool(
            config.get("access_confirmed")
        ):
            raise PermissionError(
                f"{self.name} is {self.spec.access.value}. Obtain access from "
                f"{self.spec.source} and set access_confirmed=true for local prepared data."
            )
        source_revision = str(config.get("source_revision", "")).strip()
        if not source_revision and not bool(config.get("allow_unpinned")):
            raise ValueError(
                f"{self.name} requires source_revision in the data config. "
                "Record the immutable commit, DOI release, or fixed dataset version."
            )
        if source_revision and source_revision.casefold() in {
            "main",
            "master",
            "latest",
            "head",
            "unpinned",
        }:
            raise ValueError(
                f"{self.name} source_revision must be immutable, not {source_revision!r}."
            )

    def _delegate(self, config: dict[str, Any]) -> DatasetAdapter:
        normalized = str(config.get("normalized_adapter", "")).strip().lower()
        if normalized:
            if normalized == "medical_vqa_jsonl":
                return MedicalVQADatasetAdapter()
            if normalized == "medical_tasks_jsonl":
                return MedicalTasksDatasetAdapter()
            raise ValueError(
                "normalized_adapter must be medical_vqa_jsonl or medical_tasks_jsonl."
            )
        if self.spec.adapter_family is DatasetAdapterFamily.VQA:
            return MedicalVQADatasetAdapter()
        return MedicalTasksDatasetAdapter()

    @staticmethod
    def _delegate_config(config: dict[str, Any]) -> dict[str, Any]:
        result = dict(config)
        for key in (
            "access_confirmed",
            "allow_unpinned",
            "normalized_adapter",
            "source_revision",
        ):
            result.pop(key, None)
        result["source"] = str(result.get("source", "jsonl"))
        return result

    def load(self, config: dict[str, Any], project_root: Path) -> list[Any]:
        self._validate(config)
        samples = self._delegate(config).load(self._delegate_config(config), project_root)
        for sample in samples:
            metadata = dict(getattr(sample, "metadata", {}))
            metadata.update(
                {
                    "resource": self.name,
                    "source_url": self.spec.source,
                    "source_revision": config.get("source_revision", "unpinned"),
                    "catalog_version": DATASET_RESOURCES.version,
                    "clinical_use": False,
                }
            )
            sample.metadata = metadata
        return samples

    def fingerprint(self, config: dict[str, Any], project_root: Path) -> str:
        self._validate(config)
        delegate_fingerprint = self._delegate(config).fingerprint(
            self._delegate_config(config), project_root
        )
        payload = {
            "resource": self.spec.to_dict(),
            "source_revision": config.get("source_revision", "unpinned"),
            "delegate_fingerprint": delegate_fingerprint,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()


def catalog_dataset_factory(resource_name: str):
    def create() -> CatalogDatasetAdapter:
        return CatalogDatasetAdapter(resource_name)

    return create
