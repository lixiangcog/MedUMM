from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

from medumm.core.exceptions import ConfigurationError


ENVIRONMENT_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
CURRENT_SCHEMA_VERSION = "1.0"
CONFIG_KINDS = frozenset({"inference", "evaluation", "post_training"})


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
    *,
    validate: bool = True,
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
    if validate:
        validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> str:
    """Validate the stable top-level configuration envelope.

    v0.1 files without ``schema_version`` remain readable and are interpreted as
    schema 1.0. New files should always declare it explicitly.
    """

    version = str(config.get("schema_version", CURRENT_SCHEMA_VERSION))
    if version != CURRENT_SCHEMA_VERSION:
        raise ConfigurationError(
            f"Unsupported schema_version {version!r}; expected {CURRENT_SCHEMA_VERSION!r}."
        )
    present = [kind for kind in CONFIG_KINDS if isinstance(config.get(kind), dict)]
    if len(present) > 1:
        raise ConfigurationError(
            f"A config may define one execution block, found: {', '.join(sorted(present))}."
        )
    runtime = config.get("runtime")
    if runtime is not None and not isinstance(runtime, dict):
        raise ConfigurationError("runtime must be a mapping when provided.")
    return version


def config_kind(config: dict[str, Any]) -> str:
    present = [kind for kind in CONFIG_KINDS if isinstance(config.get(kind), dict)]
    if len(present) == 1:
        return present[0]
    if "benchmark" in config:
        return "evaluation"
    if "method" in config:
        return "post_training"
    if "backbone" in config or "model" in config:
        return "inference"
    raise ConfigurationError("Unable to determine configuration kind.")


def execution_config(config: dict[str, Any], kind: str | None = None) -> dict[str, Any]:
    """Return one canonical execution block while accepting v0.1 flat files.

    Schema 1.0 uses ``inference``, ``evaluation``, or ``post_training`` as a
    single top-level envelope. The merge below is intentionally limited to the
    legacy evaluation layout where runner options were nested but the benchmark,
    model, and data selections lived beside them.
    """

    selected = kind or config_kind(config)
    if selected not in CONFIG_KINDS:
        raise ConfigurationError(f"Unknown execution kind: {selected!r}.")
    nested = config.get(selected)
    if not isinstance(nested, dict):
        return dict(config)
    if selected == "evaluation" and any(
        key in config for key in ("benchmark", "data", "model")
    ):
        legacy = {
            key: value
            for key, value in config.items()
            if key not in {"schema_version", "runtime", "evaluation"}
        }
        return {**legacy, **nested}
    return dict(nested)


def find_project_root(start: str | Path) -> Path:
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return current
