import sys
import inspect
from types import ModuleType

import pytest

from medumm.backbones.llava_med import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USE_KEYWORD_STOPPING,
    LlavaMedAdapter,
)
from medumm.core import ArchitectureFamily, TaskType
from medumm.core.runtime import RuntimeContext
from tests.conftest import PROJECT_ROOT


def test_llava_med_advertises_real_model_capabilities():
    capabilities = LlavaMedAdapter.capabilities
    assert capabilities.architecture is ArchitectureFamily.AUTOREGRESSIVE
    assert capabilities.tasks == frozenset({TaskType.UNDERSTANDING})
    assert capabilities.max_images == 1
    assert not capabilities.supports_batching
    assert DEFAULT_SYSTEM_PROMPT == ""


def test_llava_med_keyword_stopping_is_opt_in():
    # Upstream model_vqa constructs a keyword criterion but does not pass it to
    # generate. The adapter follows that behavior unless explicitly configured.
    assert DEFAULT_USE_KEYWORD_STOPPING is False
    source = inspect.getsource(LlavaMedAdapter._understand_one)
    assert '"generated_tokens"' in source
    assert '"scheduler"' in source


def test_llava_med_source_path_must_contain_package(tmp_path):
    runtime = RuntimeContext.create(
        command="test",
        config_path=PROJECT_ROOT / "pyproject.toml",
        output_directory=tmp_path / "output",
    )
    with pytest.raises(FileNotFoundError, match="must contain the llava package"):
        LlavaMedAdapter().load({"source_path": str(tmp_path)}, runtime)


def test_llava_med_missing_dependency_has_actionable_error(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "llava").mkdir(parents=True)
    runtime = RuntimeContext.create(
        command="test",
        config_path=PROJECT_ROOT / "pyproject.toml",
        output_directory=tmp_path / "output",
    )
    monkeypatch.setitem(sys.modules, "llava", ModuleType("llava"))
    with pytest.raises(RuntimeError, match="official microsoft/LLaVA-Med"):
        LlavaMedAdapter().load({"source_path": str(source)}, runtime)
