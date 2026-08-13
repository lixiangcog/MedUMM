from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

from medumm.environments.specs import EnvironmentCatalog


def default_catalog_path() -> Path:
    return Path(str(files("medumm.environments").joinpath("catalog", "models.yaml")))


ENVIRONMENT_CATALOG = EnvironmentCatalog.load(default_catalog_path())


def environment_catalog() -> dict[str, Any]:
    return ENVIRONMENT_CATALOG.to_dict()
