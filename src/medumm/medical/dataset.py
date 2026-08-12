from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from medumm.core.interfaces import DatasetAdapter
from medumm.medical.data import MedicalVQASample, load_medical_vqa
from medumm.medical.tasks import MedicalTaskSample, load_medical_tasks


class MedicalVQADatasetAdapter(DatasetAdapter):
    name = "medical_vqa_jsonl"

    @staticmethod
    def _data_path(config: dict[str, Any], project_root: Path) -> Path:
        path = Path(str(config.get("path", ""))).expanduser()
        return path if path.is_absolute() else project_root / path

    def load(
        self,
        config: dict[str, Any],
        project_root: Path,
    ) -> list[MedicalVQASample]:
        return load_medical_vqa(config, project_root=project_root)

    def fingerprint(self, config: dict[str, Any], project_root: Path) -> str:
        source = self._data_path(config, project_root)
        digest = hashlib.sha256()
        digest.update(source.read_bytes())
        stable_config = {
            key: value for key, value in config.items() if key not in {"max_samples"}
        }
        digest.update(json.dumps(stable_config, sort_keys=True, default=str).encode())
        digest.update(str(config.get("max_samples", 0)).encode())
        return digest.hexdigest()


class MedicalTasksDatasetAdapter(DatasetAdapter):
    name = "medical_tasks_jsonl"

    @staticmethod
    def _data_path(config: dict[str, Any], project_root: Path) -> Path:
        path = Path(str(config.get("path", ""))).expanduser()
        return path if path.is_absolute() else project_root / path

    def load(
        self,
        config: dict[str, Any],
        project_root: Path,
    ) -> list[MedicalTaskSample]:
        return load_medical_tasks(config, project_root=project_root)

    def fingerprint(self, config: dict[str, Any], project_root: Path) -> str:
        source = self._data_path(config, project_root)
        digest = hashlib.sha256(source.read_bytes())
        stable_config = {
            key: value for key, value in config.items() if key not in {"max_samples"}
        }
        digest.update(json.dumps(stable_config, sort_keys=True, default=str).encode())
        digest.update(str(config.get("max_samples", 0)).encode())
        return digest.hexdigest()
