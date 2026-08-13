from __future__ import annotations

import json
from pathlib import Path

import pytest

from medumm.core.builtins import register_builtins
from medumm.core.config import load_config
from medumm.core.registry import registry
from medumm.post_training import PostTrainingRunner
from medumm.post_training.acceptance import (
    ROUTE_ACCEPTANCE_ORDER,
    build_contract_smoke_config,
    verify_research_routes,
)
from medumm.post_training.research_routes import (
    ROUTES,
    _launcher_command,
    route_template,
)
from tests.conftest import PROJECT_ROOT


def _source() -> dict[str, object]:
    return {
        "repository": "",
        "revision": "d34cd2cce472fd5f2279c471c6a45ac48080b9dc",
        "license": "Apache-2.0",
        "code_root": str(PROJECT_ROOT),
        "strict_git_revision": False,
    }


def _data() -> dict[str, object]:
    return {
        "contract_manifest": "examples/medical/post_training/research_routes_contract.jsonl",
        "provenance": "examples/medical/post_training/provenance.json",
        "license": "CC0-1.0",
        "deidentified": True,
    }


def _config(method: str, output: Path) -> dict[str, object]:
    source = _source()
    source["repository"] = "https://github.com/lixiangcog/MedUMM"
    stages = []
    for index, stage in enumerate(ROUTES[method].stages):
        checkpoint = output / f"{index:02d}-{stage.name}.json"
        item: dict[str, object] = {
            "name": stage.name,
            "launcher": {
                "entrypoint": "python",
                "cwd": str(PROJECT_ROOT),
                "module": "medumm.post_training.contract_smoke_worker",
                "flag_style": "underscore",
                "bool_style": "value",
                "args": {
                    "method": method,
                    "stage": stage.name,
                    "checkpoint_path": str(checkpoint),
                    "steps": 2,
                },
            },
            "artifacts": {"checkpoint_path": str(checkpoint)},
        }
        if stage.depends_on:
            dependency = stage.depends_on[-1]
            reference = f"{{{{stages.{dependency}.checkpoint}}}}"
            item["inputs"] = {f"{dependency}_checkpoint": reference}
            item["launcher"]["args"]["previous_checkpoint"] = reference
        stages.append(item)
    return {
        "method": method,
        "execution": "launch",
        "acceptance_mode": "contract_smoke",
        "source": source,
        "data": _data(),
        "stages": stages,
        "output_directory": str(output),
    }


def test_all_research_routes_are_registered():
    register_builtins()
    assert set(ROUTES).issubset(registry.trainers.names())
    assert len(ROUTES) == 7


@pytest.mark.parametrize("method", ROUTE_ACCEPTANCE_ORDER)
def test_acceptance_config_preserves_each_route_stage_graph(method, tmp_path):
    config = build_contract_smoke_config(
        method,
        project_root=PROJECT_ROOT,
        output_directory=tmp_path / method,
        source_revision="d34cd2cce472fd5f2279c471c6a45ac48080b9dc",
    )["post_training"]
    assert config["method"] == method
    assert config["acceptance_mode"] == "contract_smoke"
    assert [stage["name"] for stage in config["stages"]] == [
        stage.name for stage in ROUTES[method].stages
    ]


def test_sequential_acceptance_runs_every_route_through_public_cli(tmp_path):
    summary = verify_research_routes(
        project_root=PROJECT_ROOT,
        output_directory=tmp_path / "sequential",
        source_revision="d34cd2cce472fd5f2279c471c6a45ac48080b9dc",
    )
    assert summary["status"] == "passed"
    assert summary["routes_passed"] == 7
    assert summary["stages_executed"] == 13
    assert [result["method"] for result in summary["routes"]] == list(
        ROUTE_ACCEPTANCE_ORDER
    )
    for result in summary["routes"]:
        assert result["paper_profile_plan"]["status"] == "passed"
        assert result["contract_execution"]["status"] == "passed"
        assert result["paper_runtime"]["status"] == "not_run"


@pytest.mark.parametrize("method", sorted(ROUTES))
def test_each_route_launches_real_contract_worker(method, tmp_path):
    output = tmp_path / method
    result = PostTrainingRunner().run(
        _config(method, output), config_path=PROJECT_ROOT / "pyproject.toml"
    )
    assert result.status == "completed"
    assert result.metrics["stages_completed"] == len(ROUTES[method].stages)
    assert Path(result.checkpoint_path).is_file()
    checkpoint = json.loads(Path(result.checkpoint_path).read_text())
    assert checkpoint["method"] == method
    assert checkpoint["paper_fidelity_claim"] is False
    preflight = json.loads((output / "preflight.json").read_text())
    assert all(stage["status"] == "ready" for stage in preflight["stages"])


def test_multistage_dependency_must_be_present(tmp_path):
    config = _config("irg", tmp_path / "irg")
    config["stages"] = [config["stages"][1]]
    config["stages"][0].pop("inputs")
    with pytest.raises(ValueError, match="requires earlier stage"):
        PostTrainingRunner().run(config, config_path=PROJECT_ROOT / "pyproject.toml")


def test_external_dependency_checkpoint_must_exist(tmp_path):
    config = _config("irg", tmp_path / "irg")
    config["stages"] = [config["stages"][1]]
    missing = str(tmp_path / "missing-stage-one.pt")
    config["stages"][0]["inputs"] = {"think_generate_checkpoint": missing}
    config["stages"][0]["launcher"]["args"]["previous_checkpoint"] = missing
    with pytest.raises(ValueError, match="dependency checkpoint not found"):
        PostTrainingRunner().run(config, config_path=PROJECT_ROOT / "pyproject.toml")


def test_launch_rejects_missing_medical_contract_fields(tmp_path):
    manifest = tmp_path / "invalid.jsonl"
    manifest.write_text('{"id":"x"}\n', encoding="utf-8")
    config = _config("reca", tmp_path / "reca")
    config["data"] = {**_data(), "contract_manifest": str(manifest)}
    with pytest.raises(ValueError, match="contract row 1 missing"):
        PostTrainingRunner().run(config, config_path=PROJECT_ROOT / "pyproject.toml")


def test_plan_writes_blocked_preflight_without_launching(tmp_path):
    config = route_template("unipath")["post_training"]
    config["output_directory"] = str(tmp_path / "plan")
    result = PostTrainingRunner().run(config, config_path=PROJECT_ROOT / "pyproject.toml")
    assert result.status == "planned"
    assert result.metadata["ready"] is False
    assert (tmp_path / "plan" / "preflight.json").is_file()


def test_unknown_route_stage_is_rejected(tmp_path):
    config = _config("reca", tmp_path / "reca")
    config["stages"][0]["name"] = "not_a_stage"
    with pytest.raises(ValueError, match="Unknown reca stage"):
        PostTrainingRunner().run(config, config_path=PROJECT_ROOT / "pyproject.toml")


def test_route_rejects_a_different_source_repository(tmp_path):
    config = _config("reca", tmp_path / "reca")
    config["source"]["repository"] = "https://example.invalid/not-reca"
    with pytest.raises(ValueError, match="contract_smoke source.repository must be"):
        PostTrainingRunner().run(config, config_path=PROJECT_ROOT / "pyproject.toml")


def test_contract_smoke_cannot_launch_an_arbitrary_module(tmp_path):
    config = _config("reca", tmp_path / "reca")
    config["stages"][0]["launcher"]["module"] = "arbitrary.training.module"
    with pytest.raises(ValueError, match="contract_smoke may only launch"):
        PostTrainingRunner().run(config, config_path=PROJECT_ROOT / "pyproject.toml")


def test_paper_runtime_requires_the_declared_upstream(tmp_path):
    config = _config("reca", tmp_path / "reca")
    config["acceptance_mode"] = "paper_runtime"
    with pytest.raises(ValueError, match="source.repository must be"):
        PostTrainingRunner().run(config, config_path=PROJECT_ROOT / "pyproject.toml")


def test_paper_runtime_cannot_disable_revision_check(tmp_path):
    config = _config("reca", tmp_path / "reca")
    config["acceptance_mode"] = "paper_runtime"
    config["source"]["repository"] = ROUTES["reca"].code_url
    with pytest.raises(ValueError, match="strict_git_revision=true"):
        PostTrainingRunner().run(config, config_path=PROJECT_ROOT / "pyproject.toml")


def test_source_license_placeholder_is_not_accepted(tmp_path):
    config = _config("reca", tmp_path / "reca")
    config["source"]["license"] = "VERIFY_AT_PINNED_REVISION"
    with pytest.raises(ValueError, match="verified concrete license"):
        PostTrainingRunner().run(config, config_path=PROJECT_ROOT / "pyproject.toml")


def test_callable_profile_uses_isolated_worker_and_redacts_config(tmp_path):
    path = PROJECT_ROOT / "configs/post_training/profiles/unigame.yaml"
    config = load_config(path)
    config["post_training"]["output_directory"] = str(tmp_path / "unigame")
    PostTrainingRunner().run(
        config["post_training"], config_path=PROJECT_ROOT / "pyproject.toml"
    )
    preflight = json.loads((tmp_path / "unigame/preflight.json").read_text())
    command = preflight["stages"][0]["command"]
    assert "medumm.post_training.callable_worker" in command
    assert "<redacted-config>" in command
    assert "UNIGAME_MEDICAL_DATASET_PATH" not in preflight["stages"][0]["command_text"]


def test_callable_worker_rejects_non_callable_targets(tmp_path):
    command = _launcher_command(
        {
            "entrypoint": "callable",
            "callable": "medumm.post_training.research_routes:ROUTES",
            "args": {},
        },
        tmp_path,
    )
    completed = __import__("subprocess").run(
        command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    assert completed.returncode != 0
    assert "Target is not callable" in completed.stderr


@pytest.mark.parametrize("method", sorted(ROUTES))
def test_pinned_route_profiles_parse_and_plan(method, tmp_path):
    path = PROJECT_ROOT / "configs" / "post_training" / "profiles" / f"{method}.yaml"
    config = load_config(path)
    config["post_training"]["output_directory"] = str(tmp_path / method)
    result = PostTrainingRunner().run(
        config["post_training"], config_path=PROJECT_ROOT / "pyproject.toml"
    )
    assert result.status == "planned"
    assert result.metadata["ready"] is False
