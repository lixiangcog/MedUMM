import pytest

from medumm.core import ArchitectureFamily, Modality, ModelCapabilities, TaskType


def test_capabilities_are_serializable_and_enforced():
    capabilities = ModelCapabilities(
        tasks=frozenset({TaskType.UNDERSTANDING}),
        input_modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
        output_modalities=frozenset({Modality.TEXT}),
        architecture=ArchitectureFamily.AUTOREGRESSIVE,
        supports_batching=True,
        max_batch_size=4,
        max_images=2,
    )
    assert capabilities.supports("understanding")
    assert capabilities.to_dict()["architecture"] == "autoregressive"
    with pytest.raises(ValueError, match="exceeds"):
        capabilities.validate_batch_size(5)


def test_capabilities_require_a_task():
    with pytest.raises(ValueError, match="at least one task"):
        ModelCapabilities(
            tasks=frozenset(),
            input_modalities=frozenset({Modality.TEXT}),
            output_modalities=frozenset({Modality.TEXT}),
            architecture=ArchitectureFamily.REFERENCE,
        )


def test_capabilities_reject_inconsistent_batch_declaration():
    with pytest.raises(ValueError, match="without batching support"):
        ModelCapabilities(
            tasks=frozenset({TaskType.UNDERSTANDING}),
            input_modalities=frozenset({Modality.TEXT}),
            output_modalities=frozenset({Modality.TEXT}),
            architecture=ArchitectureFamily.REFERENCE,
            supports_batching=False,
            max_batch_size=2,
        )
