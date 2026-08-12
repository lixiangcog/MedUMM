import json

import pytest

from medumm.medical.alignment import (
    AlignmentObjective,
    deterministic_epoch_samples,
    load_alignment_data,
)
from tests.conftest import PROJECT_ROOT


def _config():
    return {
        "path": "examples/medical/alignment/tiny_preferences.jsonl",
        "provenance": "examples/medical/alignment/provenance.json",
        "deidentified": True,
        "license": "CC0-1.0",
        "require_provenance": True,
        "require_preference_provenance": True,
    }


def test_load_alignment_data_audits_preference_provenance():
    bundle = load_alignment_data(
        _config(), project_root=PROJECT_ROOT, objective=AlignmentObjective.DPO
    )
    assert len(bundle.samples) == 4
    assert bundle.audit["status"] == "warning"
    assert bundle.audit["preference_pair_count"] == 4
    assert bundle.audit["preference_provenance_count"] == 4
    assert bundle.audit["clinical_relevance"]["maximum"] == 2.0
    assert bundle.audit["clinician_or_expert_count"] == 0
    assert bundle.fingerprint == bundle.audit["dataset_fingerprint"]


def test_alignment_requires_rejected_response(tmp_path):
    path = tmp_path / "sft.jsonl"
    path.write_text(
        json.dumps({"id": "a", "prompt": "p", "chosen": "c"}) + "\n",
        encoding="utf-8",
    )
    config = {
        "path": str(path),
        "license": "CC0-1.0",
        "deidentified": True,
        "require_provenance": False,
        "require_preference_provenance": False,
    }
    sft = load_alignment_data(
        config, project_root=PROJECT_ROOT, objective=AlignmentObjective.SFT
    )
    assert len(sft.samples) == 1
    with pytest.raises(ValueError, match="requires rejected"):
        load_alignment_data(
            config, project_root=PROJECT_ROOT, objective=AlignmentObjective.DPO
        )


def test_weighted_mixture_sampling_is_deterministic():
    bundle = load_alignment_data(
        _config(), project_root=PROJECT_ROOT, objective=AlignmentObjective.DPO
    )
    first = deterministic_epoch_samples(bundle.samples, seed=7, epoch=2, epoch_size=9)
    second = deterministic_epoch_samples(bundle.samples, seed=7, epoch=2, epoch_size=9)
    third = deterministic_epoch_samples(bundle.samples, seed=7, epoch=3, epoch_size=9)
    assert [sample.sample_id for sample in first] == [sample.sample_id for sample in second]
    assert [sample.sample_id for sample in first] != [sample.sample_id for sample in third]


def test_alignment_fingerprint_does_not_depend_on_absolute_project_path(tmp_path):
    copied = tmp_path / "copy"
    copied.mkdir()
    manifest = PROJECT_ROOT / "examples/medical/alignment/tiny_preferences.jsonl"
    provenance = PROJECT_ROOT / "examples/medical/alignment/provenance.json"
    (copied / "tiny_preferences.jsonl").write_bytes(manifest.read_bytes())
    (copied / "provenance.json").write_bytes(provenance.read_bytes())
    original = load_alignment_data(
        _config(), project_root=PROJECT_ROOT, objective=AlignmentObjective.DPO
    )
    relocated = load_alignment_data(
        {
            **_config(),
            "path": str(copied / "tiny_preferences.jsonl"),
            "provenance": str(copied / "provenance.json"),
        },
        project_root=tmp_path,
        objective=AlignmentObjective.DPO,
    )
    assert original.fingerprint == relocated.fingerprint


def test_alignment_mixture_namespaces_ids_and_uses_source_weights(tmp_path):
    source_a = tmp_path / "a.jsonl"
    source_b = tmp_path / "b.jsonl"
    record = {
        "id": "shared-id",
        "prompt": "p",
        "chosen": "chosen",
        "rejected": "rejected",
        "preference_rationale": "chosen is preferred",
        "preference_provenance": {"kind": "test"},
    }
    source_a.write_text(json.dumps(record) + "\n", encoding="utf-8")
    source_b.write_text(json.dumps(record) + "\n", encoding="utf-8")
    config = {
        "mixtures": [
            {
                "name": "a",
                "path": str(source_a),
                "weight": 1,
                "license": "CC0-1.0",
                "deidentified": True,
            },
            {
                "name": "b",
                "path": str(source_b),
                "weight": 9,
                "license": "CC0-1.0",
                "deidentified": True,
            },
        ],
        "require_provenance": False,
        "require_preference_provenance": True,
        "require_preference_rationale": True,
    }
    bundle = load_alignment_data(
        config, project_root=tmp_path, objective=AlignmentObjective.DPO
    )
    assert {sample.sample_id for sample in bundle.samples} == {
        "a:shared-id",
        "b:shared-id",
    }
    selected = deterministic_epoch_samples(
        bundle.samples, seed=2, epoch=0, epoch_size=100
    )
    assert sum(sample.source_name == "b" for sample in selected) > 75


def test_alignment_contract_preserves_medical_image_fields(tmp_path):
    image = tmp_path / "scan.png"
    image.write_bytes(b"not decoded by the data contract")
    manifest = tmp_path / "multimodal.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "image-pair",
                "prompt": "Describe the finding.",
                "chosen": "Grounded answer.",
                "rejected": "Ungrounded answer.",
                "image": image.name,
                "modality": "radiograph",
                "anatomy": "chest",
                "preference_rationale": "The chosen answer is grounded.",
                "preference_provenance": {"kind": "test"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    bundle = load_alignment_data(
        {
            "path": str(manifest),
            "image_root": str(tmp_path),
            "license": "CC0-1.0",
            "deidentified": True,
            "require_provenance": False,
        },
        project_root=tmp_path,
        objective=AlignmentObjective.DPO,
    )
    assert bundle.samples[0].image_paths == [str(image)]
    assert bundle.samples[0].modality == "radiograph"
    assert bundle.samples[0].anatomy == "chest"
    assert bundle.audit["image_sample_count"] == 1
