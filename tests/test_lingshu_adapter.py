import pytest

from medumm.backbones.lingshu import DEFAULT_SYSTEM_PROMPT, LingshuAdapter
from medumm.core import ArchitectureFamily, Modality, TaskType
from medumm.core.runtime import RuntimeContext
from tests.conftest import PROJECT_ROOT


def test_lingshu_advertises_native_medical_vlm_capabilities():
    capabilities = LingshuAdapter.capabilities
    assert capabilities.architecture is ArchitectureFamily.AUTOREGRESSIVE
    assert capabilities.tasks == frozenset({TaskType.UNDERSTANDING})
    assert Modality.IMAGE_SET in capabilities.input_modalities
    assert capabilities.max_images == 4
    assert "clinical advice" in DEFAULT_SYSTEM_PROMPT


@pytest.mark.parametrize("revision", ["", "main", "latest", "HEAD"])
def test_lingshu_requires_immutable_revision_before_importing_heavy_runtime(
    tmp_path, revision
):
    runtime = RuntimeContext.create(
        command="test",
        config_path=PROJECT_ROOT / "pyproject.toml",
        output_directory=tmp_path / "output",
    )
    with pytest.raises(ValueError, match="immutable source commit"):
        LingshuAdapter().load({"revision": revision}, runtime)


def test_lingshu_rejects_more_than_four_images():
    adapter = LingshuAdapter()
    request = type("Request", (), {"images": ["image.png"] * 5, "prompt": "question"})()
    with pytest.raises(ValueError, match="at most 4"):
        adapter._messages(request)
