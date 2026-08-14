from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from medumm.cli.main import main
from medumm.environments import ENVIRONMENT_CATALOG, EnvironmentCatalog
from medumm.environments.render import (
    render_apptainer,
    render_container,
    write_generated_artifacts,
)
from medumm.environments.specs import EnvironmentSpec
from medumm.resources import MODEL_RESOURCES
from tests.conftest import PROJECT_ROOT


GENERATED_ROOT = PROJECT_ROOT / "environments" / "models"


def test_every_model_has_exactly_one_environment_contract():
    assert set(ENVIRONMENT_CATALOG.names()) == set(MODEL_RESOURCES.names())
    assert len(ENVIRONMENT_CATALOG.values()) == 32


def test_contracts_are_immutable_and_access_aware():
    for spec in ENVIRONMENT_CATALOG.values():
        assert "@sha256:" in spec.docker_base_image
        assert "@sha256:" in spec.apptainer_base_image
        assert len(spec.model_revision) == 40
        assert all(len(source.revision) == 40 for source in spec.sources)
        assert all("==" in item or " @ " in item or "git+" in item for item in spec.dependencies)
        assert spec.access == MODEL_RESOURCES.get(spec.model).access.value
        assert spec.resolution == "uv 0.8.11; CPython 3.10; x86_64-manylinux_2_28"
        assert len(spec.fingerprint()) == 64


def test_runtime_claims_require_committed_evidence():
    runtime = [spec for spec in ENVIRONMENT_CATALOG.values() if spec.validation.value == "runtime_validated"]
    assert {spec.model for spec in runtime} == {
        "biomedclip",
        "llava_med_v1_5_7b",
        "lingshu_7b",
        "medvlm_r1",
        "plip",
        "pubmedclip",
        "quiltnet",
    }
    for spec in runtime:
        assert spec.evidence
        assert (PROJECT_ROOT / spec.evidence).is_file()


def test_generated_artifacts_are_current_and_complete():
    result = write_generated_artifacts(ENVIRONMENT_CATALOG, GENERATED_ROOT, check=True)
    assert result == {"models": 32, "artifacts": 160, "changed": [], "valid": True}
    for model in ENVIRONMENT_CATALOG.names():
        directory = GENERATED_ROOT / model
        assert {path.name for path in directory.iterdir()} == {
            "Dockerfile",
            "apptainer.def",
            "requirements.txt",
            "lock.txt",
            "sources.lock",
        }


def test_generated_shell_and_container_files_are_parseable():
    scripts = (
        PROJECT_ROOT / "scripts/setup_model_env.sh",
        PROJECT_ROOT / "scripts/build_model_container.sh",
        PROJECT_ROOT / "scripts/slurm_model_environment.sh",
    )
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)
    for spec in ENVIRONMENT_CATALOG.values():
        dockerfile = render_container(spec)
        apptainer = render_apptainer(spec)
        assert dockerfile.startswith("# syntax=docker/dockerfile:1.7")
        assert f"org.medumm.model={spec.model}" in dockerfile
        assert apptainer.startswith("# Generated")
        assert "%post" in apptainer


def test_container_builder_exposes_non_root_apptainer_modes():
    script = (PROJECT_ROOT / "scripts/build_model_container.sh").read_text(encoding="utf-8")
    assert "MEDUMM_APPTAINER_BUILD_MODE" in script
    assert "fakeroot" in script and "remote" in script and "sudo" in script


def test_modal_uses_the_same_fully_resolved_lockfiles():
    source = (PROJECT_ROOT / "modal/images.py").read_text(encoding="utf-8")
    assert "pip_install_from_requirements" in source
    assert '"lock.txt"' in source


def test_cli_lists_and_shows_environment(capsys):
    assert main(["environments", "list"]) == 0
    output = capsys.readouterr().out
    assert "llava_med_v1_5_7b: profile=llava_legacy" in output
    assert "medgemma_1_5_4b_it: profile=gemma3" in output
    assert main(["environments", "show", "lingshu_7b"]) == 0
    assert '"model": "lingshu_7b"' in capsys.readouterr().out


def test_mutable_dependency_and_image_are_rejected(tmp_path: Path):
    catalog = tmp_path / "bad.yaml"
    catalog.write_text(
        """
catalog_version: "1"
models:
  - model: unsafe
    profile: unsafe
    python: "3.10"
    platform: linux_x86_64
    accelerator: nvidia_gpu
    cuda: "12.6"
    docker_base_image: nvidia/cuda:latest
    apptainer_base_image: nvidia/cuda:latest
    dependencies: ["torch>=2"]
    sources: []
    model_revision: 0123456789012345678901234567890123456789
    minimum_gpu_memory_gb: 1
    recommended_gpus: 1
    access: open
    imports: [torch]
    validation: contract_validated
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        EnvironmentCatalog.load(catalog)


def test_model_setup_rejects_gated_assets_without_explicit_acceptance():
    result = subprocess.run(
        ["bash", "scripts/setup_model_env.sh", "medgemma_1_5_4b_it", "--check-only"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 3
    assert "--accept-terms" in result.stderr
