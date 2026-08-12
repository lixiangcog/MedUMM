import pytest

from medumm.core.exceptions import ComponentNotFoundError, DuplicateComponentError
from medumm.core.registry import ComponentHub, TypedRegistry


def test_typed_registry_lists_and_creates_components():
    registry = TypedRegistry("example")
    registry.register("sample", lambda: {"ready": True}, description="sample component")
    assert registry.names() == ["sample"]
    assert registry.create("sample") == {"ready": True}
    assert registry.descriptors()[0].description == "sample component"


def test_typed_registry_rejects_duplicates_and_unknown_names():
    registry = TypedRegistry("example")
    registry.register("sample", object)
    with pytest.raises(DuplicateComponentError):
        registry.register("sample", object)
    with pytest.raises(ComponentNotFoundError, match="available: sample"):
        registry.create("missing")


def test_component_hub_supports_stable_kind_aliases():
    hub = ComponentHub()
    hub.register("backbone", "sample", lambda: {"kind": "model"})
    hub.register("evaluator", "score", lambda: {"kind": "benchmark"})
    assert hub.names("model") == ["sample"]
    assert hub.create("benchmark", "score") == {"kind": "benchmark"}
