import json
import subprocess

import pytest

from medumm.backbones.audit import adapter_catalog, preflight_model_adapter
from medumm.backbones.catalog_model import CatalogModelAdapter
from medumm.backbones.recipes import (
    MODEL_ADAPTER_RECIPES,
    AdapterImplementation,
    ModelExecutor,
)
from medumm.cli.main import main
from medumm.core.config import load_config
from medumm.core.runtime import RuntimeContext
from medumm.environments import ENVIRONMENT_CATALOG
from medumm.resources import MODEL_RESOURCES
from tests.conftest import PROJECT_ROOT


def test_every_model_has_one_explicit_adapter_recipe_and_environment():
    expected = set(MODEL_RESOURCES.names())
    assert set(MODEL_ADAPTER_RECIPES.names()) == expected
    assert set(ENVIRONMENT_CATALOG.names()) == expected
    assert len(expected) == 32
    assert all(recipe.prompt_protocol for recipe in MODEL_ADAPTER_RECIPES.values())
    assert all(recipe.model_type for recipe in MODEL_ADAPTER_RECIPES.values())


def test_official_source_recipes_are_fixed_and_pinned_not_user_bridge_placeholders():
    for recipe in MODEL_ADAPTER_RECIPES.values():
        if recipe.implementation is not AdapterImplementation.OFFICIAL_SOURCE:
            continue
        assert recipe.source_checkout_required is True
        assert recipe.official_entrypoint
        assert "REPLACE" not in recipe.official_entrypoint
        assert ENVIRONMENT_CATALOG.get(recipe.name).sources


def test_adapter_audit_distinguishes_recipes_from_gpu_evidence():
    result = adapter_catalog()
    assert result["valid"] is True
    assert result["models"] == 32
    assert result["explicit_recipes"] == 32
    assert result["builtin_executors"] == 18
    assert result["official_source_executors"] == 14
    assert result["runtime_validated"] == 7


def test_cli_exposes_model_adapter_matrix(capsys):
    assert main(["models", "audit"]) == 0
    audit = json.loads(capsys.readouterr().out)
    assert audit["valid"] is True
    assert audit["explicit_recipes"] == 32

    assert main(["models", "show", "medvlm_r1"]) == 0
    recipe = json.loads(capsys.readouterr().out)
    assert recipe["executor"] == ModelExecutor.QWEN2_VL.value
    assert recipe["model_class"] == "Qwen2VLForConditionalGeneration"


def test_templates_never_ask_for_arbitrary_python_bridge(capsys):
    assert main(["resources", "template", "bimedix2_8b", "--kind", "model"]) == 0
    template = capsys.readouterr().out
    assert "bridge:" not in template
    assert "source_path: REPLACE_WITH_PINNED_UPSTREAM_CHECKOUT" in template


def test_catalog_adapter_dispatches_by_real_model_recipe(monkeypatch, tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    adapter = CatalogModelAdapter("medvlm_r1")
    called = {}

    def fake_load(config, *, model_path, revision):
        called.update(config=config, model_path=model_path, revision=revision)

    monkeypatch.setattr(adapter, "_load_qwen_vl", fake_load)
    runtime = RuntimeContext.create(
        command="test",
        config_path=PROJECT_ROOT / "configs" / "inference" / "reference_understanding.yaml",
        output_directory=tmp_path / "run",
    )
    adapter.load(
        {"model_path": str(model_path), "revision": "d256f2cfdf98c6872c1dc9f20b7dd52f49374fe9"},
        runtime,
    )
    assert called["model_path"] == str(model_path)
    assert called["revision"] == "d256f2cfdf98c6872c1dc9f20b7dd52f49374fe9"


def test_preflight_fails_closed_on_revision_and_missing_assets(tmp_path):
    result = preflight_model_adapter(
        "plip",
        project_root=PROJECT_ROOT,
        model_path=tmp_path / "missing",
        revision="main",
        check_imports=False,
    )
    assert result["ready"] is False
    assert {item["check"] for item in result["checks"] if not item["passed"]} == {
        "model_revision",
        "model_path",
    }


def test_official_runtime_failure_is_owned_by_medumm_not_user_bridge(tmp_path):
    source = tmp_path / "source"
    model = tmp_path / "model"
    source.mkdir()
    model.mkdir()
    adapter = CatalogModelAdapter("radfm_14b")
    runtime = RuntimeContext.create(
        command="test",
        config_path=PROJECT_ROOT / "configs" / "inference" / "reference_understanding.yaml",
        output_directory=tmp_path / "run",
    )
    with pytest.raises(RuntimeError, match="adapter implementation failure"):
        adapter.load(
            {
                "model_path": str(model),
                "revision": "bd5e695420b1104e0663401c6c1012bf9f29e87e",
                "source_path": str(source),
            },
            runtime,
        )


def test_real_adapter_acceptance_configs_select_explicit_recipes(monkeypatch):
    image = PROJECT_ROOT / "examples" / "medical" / "images" / "synthetic_scan.pgm"
    monkeypatch.setenv("MEDUMM_ADAPTER_SMOKE_IMAGE", str(image))
    values = {
        "plip": ("PLIP_MODEL_PATH", "plip"),
        "quiltnet": ("QUILTNET_MODEL_PATH", "quiltnet"),
        "medvlm_r1": ("MEDVLM_R1_MODEL_PATH", "medvlm-r1"),
        "biomedclip": ("BIOMEDCLIP_MODEL_PATH", "biomedclip"),
    }
    monkeypatch.setenv("BIOMEDCLIP_TEXT_MODEL_PATH", "/models/biomedclip-text-model")
    for model, (variable, directory) in values.items():
        monkeypatch.setenv(variable, f"/models/{directory}")
        config = load_config(PROJECT_ROOT / "configs" / "inference" / f"{model}_v1.4.yaml")
        block = config["inference"]
        assert block["backbone"] == model
        assert block["config"]["revision"] == ENVIRONMENT_CATALOG.get(model).model_revision
        assert MODEL_ADAPTER_RECIPES.get(model).implementation is AdapterImplementation.BUILTIN


def test_v14_acceptance_scripts_are_parseable_and_use_isolated_environments():
    scripts = (
        "slurm_prepare_real_model_assets_v1.4.sh",
        "slurm_prepare_model_envs_v1.4.sh",
        "slurm_real_model_adapters_v1.4.sh",
    )
    for name in scripts:
        subprocess.run(["bash", "-n", str(PROJECT_ROOT / "scripts" / name)], check=True)
    runtime = (PROJECT_ROOT / "scripts" / scripts[-1]).read_text(encoding="utf-8")
    assert '"${MEDUMM_ENV_ROOT}/plip/bin/python"' in runtime
    assert '"${MEDUMM_ENV_ROOT}/medvlm_r1/bin/python"' in runtime
    assert "HF_HUB_OFFLINE=1" in runtime
