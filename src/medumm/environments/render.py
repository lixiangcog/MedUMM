from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from medumm.environments.specs import EnvironmentCatalog, EnvironmentSpec


def render_requirements(spec: EnvironmentSpec) -> str:
    return spec.requirements()


def render_container(spec: EnvironmentSpec) -> str:
    index = f" --extra-index-url {spec.torch_index}" if spec.torch_index else ""
    lines = [
        "# syntax=docker/dockerfile:1.7",
        "# Generated from environments/models.yaml; do not edit by hand.",
        f"FROM {spec.docker_base_image}",
        "ARG DEBIAN_FRONTEND=noninteractive",
        "RUN apt-get update && apt-get install -y --no-install-recommends \\",
        f"    git python{spec.python} python{spec.python}-venv python3-pip ca-certificates \\",
        "    && rm -rf /var/lib/apt/lists/*",
        f"RUN python{spec.python} -m venv /opt/medumm/venv",
        "ENV PATH=/opt/medumm/venv/bin:$PATH \\",
        "    PYTHONUNBUFFERED=1 \\",
        "    PIP_DISABLE_PIP_VERSION_CHECK=1 \\",
        "    HF_HOME=/cache/huggingface \\",
        "    MEDUMM_MODEL_ROOT=/models \\",
        "    MEDUMM_OUTPUT_ROOT=/outputs",
        "COPY lock.txt /tmp/medumm-lock.txt",
        "RUN python -m pip install --upgrade pip==25.1.1 \\",
        f"    && python -m pip install{index} -r /tmp/medumm-lock.txt",
        "WORKDIR /workspace",
        f"LABEL org.medumm.model={spec.model} \\",
        f"      org.medumm.profile={spec.profile} \\",
        f"      org.medumm.contract-sha256={spec.fingerprint()}",
        "CMD [\"python\", \"-m\", \"medumm.cli.main\", \"--help\"]",
        "",
    ]
    return "\n".join(lines)


def render_apptainer(spec: EnvironmentSpec) -> str:
    return "\n".join(
        [
            "# Generated from environments/models.yaml; do not edit by hand.",
            "Bootstrap: docker",
            f"From: {spec.apptainer_base_image}",
            "",
            "%files",
            "    lock.txt /tmp/medumm-lock.txt",
            "",
            "%post",
            "    apt-get update",
            f"    apt-get install -y --no-install-recommends git python{spec.python} python{spec.python}-venv python3-pip ca-certificates",
            "    rm -rf /var/lib/apt/lists/*",
            f"    python{spec.python} -m venv /opt/medumm/venv",
            "    /opt/medumm/venv/bin/python -m pip install --upgrade pip==25.1.1",
            "    "
            + (
                f"/opt/medumm/venv/bin/python -m pip install "
                f"{'--extra-index-url ' + spec.torch_index + ' ' if spec.torch_index else ''}"
                "-r /tmp/medumm-lock.txt"
            ),
            "",
            "%environment",
            "    export PATH=/opt/medumm/venv/bin:$PATH",
            "    export PYTHONUNBUFFERED=1",
            "    export HF_HOME=/cache/huggingface",
            "    export MEDUMM_MODEL_ROOT=/models",
            "    export MEDUMM_OUTPUT_ROOT=/outputs",
            "",
            "%labels",
            f"    org.medumm.model {spec.model}",
            f"    org.medumm.profile {spec.profile}",
            f"    org.medumm.contract-sha256 {spec.fingerprint()}",
            "",
        ]
    )


def write_generated_artifacts(
    catalog: EnvironmentCatalog,
    output_root: str | Path,
    *,
    check: bool = False,
) -> dict[str, Any]:
    root = Path(output_root)
    changed: list[str] = []
    expected: set[Path] = set()
    for spec in catalog.values():
        directory = root / spec.model
        artifacts = {
            directory / "requirements.txt": render_requirements(spec),
            directory / "lock.txt": _read_resolved_lock(root, spec),
            directory / "Dockerfile": render_container(spec),
            directory / "apptainer.def": render_apptainer(spec),
            directory / "sources.lock": spec.source_manifest(),
        }
        for path, content in artifacts.items():
            expected.add(path)
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                changed.append(str(path))
                if not check:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")
    existing = (
        set(root.glob("*/requirements.txt"))
        | set(root.glob("*/Dockerfile"))
        | set(root.glob("*/apptainer.def"))
        | set(root.glob("*/sources.lock"))
        | set(root.glob("*/lock.txt"))
    )
    stale = sorted(str(path) for path in existing - expected)
    if stale:
        changed.extend(stale)
        if not check:
            for path in existing - expected:
                path.unlink()
    return {
        "models": len(catalog.values()),
        "artifacts": len(expected),
        "changed": sorted(changed),
        "valid": not changed if check else True,
    }


def _read_resolved_lock(generated_root: Path, spec: EnvironmentSpec) -> str:
    lock = generated_root.parent / "locks" / f"{spec.model}.txt"
    if not lock.is_file():
        raise FileNotFoundError(
            f"Resolved lock is missing for {spec.model}: run scripts/resolve_model_environments.py"
        )
    content = lock.read_text(encoding="utf-8")
    header = (
        "# Generated by uv from the per-model input contract.\n"
        f"# model={spec.model}\n"
        f"# contract_sha256={spec.fingerprint()}\n"
    )
    while content.startswith("#"):
        _, _, content = content.partition("\n")
    return header + content


def inspect_current_environment(spec: EnvironmentSpec) -> dict[str, Any]:
    installed: dict[str, str | None] = {}
    for dependency in spec.dependencies:
        package = _package_name(dependency)
        if package in installed:
            continue
        try:
            installed[package] = version(package)
        except PackageNotFoundError:
            installed[package] = None
    imports: dict[str, bool] = {}
    for name in spec.imports:
        process = subprocess.run(
            [sys.executable, "-c", f"import {name}"],
            capture_output=True,
            check=False,
            text=True,
        )
        imports[name] = process.returncode == 0
    return {
        "model": spec.model,
        "contract_sha256": spec.fingerprint(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": installed,
        "imports": imports,
        "valid": all(imports.values()),
    }


def _package_name(requirement: str) -> str:
    if "git+" in requirement or requirement.startswith(("http://", "https://")):
        return requirement.split("#egg=", 1)[-1].split(" @ ", 1)[0]
    for separator in ("===", "==", " @ "):
        if separator in requirement:
            return requirement.split(separator, 1)[0].split("[", 1)[0].strip()
    return requirement.split("[", 1)[0].strip()


def available_container_engine() -> str | None:
    for executable in ("docker", "podman", "apptainer", "singularity"):
        if shutil.which(executable):
            return executable
    return None


def write_json_report(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
