import pytest

from medumm.core.config import config_kind, execution_config, load_config
from medumm.core.exceptions import ConfigurationError


def test_yaml_override_and_environment(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  path: ${MODEL_PATH}\n  count: 1\n", encoding="utf-8")
    monkeypatch.setenv("MODEL_PATH", "/models/example")
    config = load_config(config_path, ["model.count=3", "enabled=true"])
    assert config == {"model": {"path": "/models/example", "count": 3}, "enabled": True}


def test_execution_config_unifies_new_and_legacy_evaluation_shapes():
    canonical = {
        "schema_version": "1.0",
        "evaluation": {"benchmark": "medical_vqa", "mode": "score"},
    }
    legacy = {
        "benchmark": "medical_vqa",
        "model": {"backbone": "medical_reference"},
        "evaluation": {"mode": "score"},
    }
    assert config_kind(canonical) == "evaluation"
    assert execution_config(canonical) == canonical["evaluation"]
    assert execution_config(legacy)["benchmark"] == "medical_vqa"
    assert execution_config(legacy)["mode"] == "score"


def test_config_rejects_multiple_execution_blocks(tmp_path):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        'schema_version: "1.0"\ninference: {}\npost_training: {}\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="one execution block"):
        load_config(config_path)
