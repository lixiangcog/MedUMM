from medumm.core.config import load_config


def test_yaml_override_and_environment(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  path: ${MODEL_PATH}\n  count: 1\n", encoding="utf-8")
    monkeypatch.setenv("MODEL_PATH", "/models/example")
    config = load_config(config_path, ["model.count=3", "enabled=true"])
    assert config == {"model": {"path": "/models/example", "count": 3}, "enabled": True}
