from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


ENVIRONMENT_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        return ENVIRONMENT_PATTERN.sub(
            lambda match: os.environ.get(match.group(1), match.group(0)), value
        )
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def _set_value(config: dict[str, Any], assignment: str) -> None:
    if "=" not in assignment:
        raise ValueError(f"Invalid override {assignment!r}; expected key=value.")
    dotted_key, raw_value = assignment.split("=", 1)
    parts = [part for part in dotted_key.strip().split(".") if part]
    if not parts:
        raise ValueError("Configuration override has an empty key.")
    cursor = config
    for part in parts[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot descend into non-mapping key {part!r}.")
        cursor = child
    cursor[parts[-1]] = yaml.safe_load(raw_value)


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Load a YAML or JSON mapping and apply dotted ``key=value`` overrides."""

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8") as reader:
        if source.suffix.lower() in {".yaml", ".yml"}:
            loaded = yaml.safe_load(reader)
        elif source.suffix.lower() == ".json":
            loaded = json.load(reader)
        else:
            raise ValueError(f"Unsupported configuration format: {source.suffix}")
    if loaded is None:
        config: dict[str, Any] = {}
    elif isinstance(loaded, dict):
        config = _expand_environment(loaded)
    else:
        raise ValueError("The top level of a configuration must be a mapping.")
    for assignment in overrides or []:
        _set_value(config, assignment)
    return config


def find_project_root(start: str | Path) -> Path:
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return current
