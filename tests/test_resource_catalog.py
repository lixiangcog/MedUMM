import json

import pytest

from medumm.cli.main import main
from medumm.core.builtins import register_builtins
from medumm.core.registry import registry
from medumm.medical.catalog_dataset import CatalogDatasetAdapter
from medumm.resources import (
    AccessLevel,
    DATASET_RESOURCES,
    MODEL_RESOURCES,
    IntegrationStatus,
    resource_catalog,
)
from tests.conftest import PROJECT_ROOT


def test_scale_catalog_exceeds_twenty_models_and_datasets():
    assert len(MODEL_RESOURCES.values()) >= 20
    assert len(DATASET_RESOURCES.values()) >= 20


def test_runtime_validated_status_is_reserved_for_committed_acceptance_paths():
    assert {
        item.name
        for item in MODEL_RESOURCES.values()
        if item.status is IntegrationStatus.RUNTIME_VALIDATED
    } == {"llava_med_v1_5_7b", "lingshu_7b", "pubmedclip"}
    assert {
        item.name
        for item in DATASET_RESOURCES.values()
        if item.status is IntegrationStatus.RUNTIME_VALIDATED
    } == {"vqa_rad", "slake", "path_vqa", "pneumoniamnist"}
    assert len(MODEL_RESOURCES.names()) == len(set(MODEL_RESOURCES.names()))
    assert len(DATASET_RESOURCES.names()) == len(set(DATASET_RESOURCES.names()))


def test_every_resource_has_auditable_source_and_support_metadata():
    for item in (*MODEL_RESOURCES.values(), *DATASET_RESOURCES.values()):
        assert item.source.startswith("https://")
        assert item.paper.startswith("https://")
        assert item.official_code is None or item.official_code.startswith("https://")
        assert item.license
        assert item.access in AccessLevel
        assert item.status in IntegrationStatus
        assert item.revision_policy
        assert item.medical_domains


def test_every_resource_is_registered_as_an_individual_plugin():
    register_builtins()
    assert set(MODEL_RESOURCES.names()) <= set(registry.models.names())
    assert set(DATASET_RESOURCES.names()) <= set(registry.datasets.names())
    for name in MODEL_RESOURCES.names():
        adapter = registry.models.create(name)
        assert adapter.name == name
        assert adapter.capabilities.tasks
    for name in DATASET_RESOURCES.names():
        assert registry.datasets.create(name).name == name


def test_open_dataset_adapter_loads_normalized_manifest_with_provenance():
    adapter = CatalogDatasetAdapter("vqa_rad")
    config = {
        "path": "examples/medical/tiny_eval.jsonl",
        "image_root": "examples/medical/images",
        "source_revision": "0123456789abcdef",
    }
    samples = adapter.load(config, PROJECT_ROOT)
    assert samples[0].metadata["resource"] == "vqa_rad"
    assert samples[0].metadata["source_revision"] == "0123456789abcdef"
    assert len(adapter.fingerprint(config, PROJECT_ROOT)) == 64


def test_pathvqa_uses_public_catalog_resource_name():
    register_builtins()
    assert "path_vqa" in registry.datasets.names()
    assert registry.datasets.create("path_vqa").name == "path_vqa"


def test_resource_dataset_requires_source_pin_and_access_confirmation():
    with pytest.raises(ValueError, match="source_revision"):
        CatalogDatasetAdapter("vqa_rad").load(
            {
                "path": "examples/medical/tiny_eval.jsonl",
                "image_root": "examples/medical/images",
            },
            PROJECT_ROOT,
        )
    with pytest.raises(ValueError, match="immutable"):
        CatalogDatasetAdapter("vqa_rad").load(
            {
                "path": "examples/medical/tiny_eval.jsonl",
                "image_root": "examples/medical/images",
                "source_revision": "main",
            },
            PROJECT_ROOT,
        )
    with pytest.raises(PermissionError, match="credentialed"):
        CatalogDatasetAdapter("mimic_cxr").load(
            {
                "path": "examples/medical/tiny_tasks.jsonl",
                "image_root": "examples/medical/images",
                "source_revision": "2.1.0",
            },
            PROJECT_ROOT,
        )


def test_resources_api_and_cli_validation(capsys):
    values = resource_catalog()
    assert values["counts"]["models"] >= 20
    assert values["counts"]["datasets"] >= 20
    assert main(["resources", "validate"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] is True
    assert report["registered_models"] == values["counts"]["models"]
    assert report["registered_datasets"] == values["counts"]["datasets"]


def test_resource_template_has_pin_and_access_gate(capsys):
    assert main(
        ["resources", "template", "medgemma_1_5_4b_it", "--kind", "model"]
    ) == 0
    text = capsys.readouterr().out
    assert "REPLACE_WITH_IMMUTABLE_COMMIT" in text
    assert "accept_terms: false" in text

    assert main(
        ["resources", "template", "llava_med_v1_5_7b", "--kind", "model"]
    ) == 0
    llava_template = capsys.readouterr().out
    assert "bridge:" not in llava_template
    assert "model_path: microsoft/llava-med-v1.5-mistral-7b" in llava_template


def test_volume_and_video_paths_survive_normalization(tmp_path):
    manifest = tmp_path / "tasks.jsonl"
    volume = tmp_path / "volume.bin"
    video = tmp_path / "video.bin"
    volume.write_bytes(b"volume")
    video.write_bytes(b"video")
    manifest.write_text(
        json.dumps(
            {
                "id": "case-media",
                "task": "diagnostic_reasoning",
                "prompt": "Assess the study.",
                "answer": "synthetic",
                "volume": volume.name,
                "video": video.name,
                "reference_provenance": {"kind": "synthetic"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    samples = CatalogDatasetAdapter("organmnist3d").load(
        {
            "path": str(manifest),
            "source_revision": "v2",
            "allow_unpinned": False,
        },
        PROJECT_ROOT,
    )
    assert samples[0].volume_paths == [str(volume)]
    assert samples[0].video_paths == [str(video)]
