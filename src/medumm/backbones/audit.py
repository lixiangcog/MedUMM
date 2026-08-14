from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from typing import Any

from medumm.backbones.recipes import (
    MODEL_ADAPTER_RECIPES,
    AdapterImplementation,
    ModelAdapterRecipe,
)
from medumm.environments import ENVIRONMENT_CATALOG
from medumm.resources import AccessLevel, IntegrationStatus, MODEL_RESOURCES


def _resolve(raw_path: str | Path | None, project_root: Path) -> Path | None:
    if raw_path is None or not str(raw_path).strip():
        return None
    path = Path(str(raw_path)).expanduser()
    return path if path.is_absolute() else project_root / path


def adapter_catalog() -> dict[str, Any]:
    resource_names = set(MODEL_RESOURCES.names())
    recipe_names = set(MODEL_ADAPTER_RECIPES.names())
    environment_names = set(ENVIRONMENT_CATALOG.names())
    rows = []
    for recipe in MODEL_ADAPTER_RECIPES.values():
        resource = MODEL_RESOURCES.get(recipe.name)
        environment = ENVIRONMENT_CATALOG.get(recipe.name)
        rows.append(
            {
                **recipe.to_dict(),
                "artifact_id": resource.artifact_id,
                "access": resource.access.value,
                "catalog_status": resource.status.value,
                "model_revision": environment.model_revision,
                "environment_profile": environment.profile,
                "environment_validation": environment.validation.value,
                "source_revisions": [
                    {
                        "repository": source.repository,
                        "revision": source.revision,
                    }
                    for source in environment.sources
                ],
                "evidence": environment.evidence,
            }
        )
    missing_recipes = sorted(resource_names - recipe_names)
    orphan_recipes = sorted(recipe_names - resource_names)
    environment_drift = sorted(resource_names ^ environment_names)
    runtime_count = sum(
        resource.status is IntegrationStatus.RUNTIME_VALIDATED
        for resource in MODEL_RESOURCES.values()
    )
    builtin_count = sum(
        recipe.implementation is AdapterImplementation.BUILTIN
        for recipe in MODEL_ADAPTER_RECIPES.values()
    )
    return {
        "schema_version": "1.0",
        "models": len(resource_names),
        "explicit_recipes": len(recipe_names),
        "builtin_executors": builtin_count,
        "official_source_executors": len(recipe_names) - builtin_count,
        "runtime_validated": runtime_count,
        "missing_recipes": missing_recipes,
        "orphan_recipes": orphan_recipes,
        "environment_drift": environment_drift,
        "valid": not missing_recipes and not orphan_recipes and not environment_drift,
        "recipes": rows,
    }

def _source_commit(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().lower() if result.returncode == 0 else None


def _required_imports(recipe: ModelAdapterRecipe) -> tuple[str, ...]:
    environment = ENVIRONMENT_CATALOG.get(recipe.name)
    return tuple(dict.fromkeys(environment.imports))


def preflight_model_adapter(
    name: str,
    *,
    project_root: Path,
    model_path: str | Path | None = None,
    source_path: str | Path | None = None,
    revision: str | None = None,
    accept_terms: bool = False,
    check_imports: bool = True,
) -> dict[str, Any]:
    recipe = MODEL_ADAPTER_RECIPES.get(name)
    resource = MODEL_RESOURCES.get(name)
    environment = ENVIRONMENT_CATALOG.get(name)
    checks: list[dict[str, Any]] = []

    def record(check: str, passed: bool, detail: str) -> None:
        checks.append({"check": check, "passed": passed, "detail": detail})

    provided_revision = str(revision or "").strip().lower()
    record(
        "model_revision",
        provided_revision == environment.model_revision,
        (
            f"expected={environment.model_revision}; provided={provided_revision or '<missing>'}"
        ),
    )
    access_passed = resource.access is AccessLevel.OPEN or accept_terms
    record(
        "access_terms",
        access_passed,
        f"access={resource.access.value}; accepted={str(accept_terms).lower()}",
    )

    resolved_model = _resolve(model_path, project_root)
    if resolved_model is None:
        record("model_path", False, "A local pinned model snapshot is required for preflight.")
    else:
        exists = resolved_model.exists()
        record("model_path", exists, str(resolved_model))
        if exists and resolved_model.is_dir():
            has_config = any(
                (resolved_model / filename).is_file()
                for filename in ("config.json", "open_clip_config.json", "pytorch_model.bin")
            )
            record(
                "model_manifest",
                has_config,
                "config.json, open_clip_config.json, or pytorch_model.bin",
            )

    resolved_source = _resolve(source_path, project_root)
    if recipe.source_checkout_required:
        source_ok = resolved_source is not None and resolved_source.is_dir()
        record(
            "source_checkout",
            source_ok,
            str(resolved_source) if resolved_source is not None else "missing source_path",
        )
        expected_commits = {source.revision for source in environment.sources}
        actual_commit = _source_commit(resolved_source) if source_ok else None
        record(
            "source_revision",
            actual_commit in expected_commits,
            (
                f"expected={','.join(sorted(expected_commits)) or '<catalog missing source>'}; "
                f"provided={actual_commit or '<unresolved>'}"
            ),
        )

    imports: list[dict[str, Any]] = []
    if check_imports:
        for module_name in _required_imports(recipe):
            available = importlib.util.find_spec(module_name) is not None
            imports.append({"module": module_name, "available": available})
        record(
            "environment_imports",
            all(item["available"] for item in imports),
            ", ".join(
                f"{item['module']}={'ok' if item['available'] else 'missing'}"
                for item in imports
            ),
        )

    return {
        "schema_version": "1.0",
        "model": recipe.name,
        "executor": recipe.executor.value,
        "implementation": recipe.implementation.value,
        "official_entrypoint": recipe.official_entrypoint,
        "expected_model_revision": environment.model_revision,
        "checks": checks,
        "imports": imports,
        "ready": all(check["passed"] for check in checks),
        "scope": "local_assets_source_revision_access_and_imports",
    }
