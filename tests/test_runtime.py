import json

from medumm.core.runtime import RuntimeContext, environment_snapshot, write_run_manifest
from tests.conftest import PROJECT_ROOT


def test_run_manifest_records_environment_and_redacts_secrets(tmp_path):
    runtime = RuntimeContext.create(
        command="test",
        config_path=PROJECT_ROOT / "pyproject.toml",
        output_directory=tmp_path,
        runtime_config={"run_id": "stable-test"},
    )
    path = write_run_manifest(
        runtime,
        config={
            "model": {
                "access_token": "do-not-write",
                "max_new_tokens": 128,
                "nested": {"password": "hidden"},
            }
        },
        component={"kind": "model", "name": "reference"},
        status="completed",
        result={"metadata": {"auth_token": "hidden-result"}},
    )
    manifest = json.loads(path.read_text())
    assert manifest["runtime"]["run_id"] == "stable-test"
    assert manifest["environment"]["python"]
    assert manifest["config"]["model"]["access_token"] == "<redacted>"
    assert manifest["config"]["model"]["max_new_tokens"] == 128
    assert manifest["config"]["model"]["nested"]["password"] == "<redacted>"
    assert manifest["result"]["metadata"]["auth_token"] == "<redacted>"


def test_distributed_manifest_has_rank_local_name(tmp_path):
    runtime = RuntimeContext.create(
        command="evaluation",
        config_path=PROJECT_ROOT / "pyproject.toml",
        output_directory=tmp_path,
    )
    runtime.rank = 1
    runtime.world_size = 2
    runtime.distributed = True
    path = write_run_manifest(
        runtime,
        config={},
        component={"kind": "benchmark", "name": "test"},
        status="completed",
    )
    assert path.name == "run_manifest.rank-00001-of-00002.json"


def test_environment_snapshot_accepts_explicit_source_commit(tmp_path, monkeypatch):
    runtime = RuntimeContext.create(
        command="test",
        config_path=tmp_path,
        output_directory=tmp_path / "out",
    )
    monkeypatch.setenv("MEDUMM_SOURCE_COMMIT", "release-source-commit")
    assert environment_snapshot(runtime)["git_commit"] == "release-source-commit"
