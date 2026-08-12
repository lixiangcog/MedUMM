import json

from medumm.core.runtime import RuntimeContext, write_run_manifest
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
