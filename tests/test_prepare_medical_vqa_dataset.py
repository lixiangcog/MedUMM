from importlib.util import module_from_spec, spec_from_file_location
import json

import pytest
from PIL import Image

from tests.conftest import PROJECT_ROOT


def _module():
    path = PROJECT_ROOT / "scripts/prepare_medical_vqa_dataset.py"
    spec = spec_from_file_location("prepare_medical_vqa_dataset", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_slake_export_filters_language_and_preserves_source_fields(tmp_path):
    module = _module()
    image_root = tmp_path / "slake-images"
    source = tmp_path / "slake-test.json"
    output = tmp_path / "normalized"
    (image_root / "xmlab1").mkdir(parents=True)
    Image.new("RGB", (6, 6), color="white").save(image_root / "xmlab1" / "source.jpg")
    source.write_text(
        json.dumps(
            [
                {
                    "qid": 11,
                    "img_id": 1,
                    "img_name": "xmlab1/source.jpg",
                    "question": "Does the picture contain liver?",
                    "answer": "Yes",
                    "q_lang": "en",
                    "modality": "MRI",
                    "location": "Abdomen",
                    "answer_type": "CLOSED",
                    "content_type": "Organ",
                },
                {
                    "qid": 12,
                    "img_id": 1,
                    "img_name": "xmlab1/source.jpg",
                    "question": "图像中有肝脏吗？",
                    "answer": "是",
                    "q_lang": "zh",
                },
            ]
        ),
        encoding="utf-8",
    )
    result = module.export_medical_vqa_dataset(
        dataset="slake",
        revision="a9083ce6c34ac3ffb17671a605962924d8a8f9e9",
        split="test",
        output_directory=output,
        max_samples=4,
        language="en",
        closed_only=False,
        source_path=source,
        image_root=image_root,
    )
    record = json.loads((output / "samples.jsonl").read_text())
    assert result["sample_count"] == 1
    assert result["skipped"]["language"] == 1
    assert record["answer_type"] == "closed"
    assert record["modality"] == "mri"
    assert record["category"] == "Organ"
    assert record["metadata"]["source_qid"] == 11
    assert len(record["metadata"]["source_image_sha256"]) == 64


def test_pathvqa_export_accepts_embedded_images(tmp_path):
    module = _module()
    source = tmp_path / "pathvqa.json"
    image_root = tmp_path / "images"
    output = tmp_path / "normalized"
    image_root.mkdir()
    Image.new("RGB", (6, 6), color="pink").save(image_root / "case.png")
    source.write_text(
        json.dumps(
            [
                {
                    "image_name": "case.png",
                    "question": "What tissue is shown?",
                    "answer": "colon",
                }
            ]
        ),
        encoding="utf-8",
    )
    result = module.export_medical_vqa_dataset(
        dataset="pathvqa",
        revision="c9602cf70e5c15df6e2ec2d33e33e5a70950e2d3",
        split="test",
        output_directory=output,
        max_samples=1,
        source_path=source,
        image_root=image_root,
    )
    record = json.loads((output / "samples.jsonl").read_text())
    assert result["sample_count"] == 1
    assert record["modality"] == "pathology"
    assert record["answer_type"] == "open"


def test_export_requires_immutable_revision(tmp_path):
    module = _module()
    with pytest.raises(ValueError, match="immutable"):
        module.export_medical_vqa_dataset(
            dataset="slake",
            revision="main",
            split="test",
            output_directory=tmp_path,
            source_path=tmp_path / "missing.json",
            image_root=tmp_path,
        )


def test_pathvqa_export_enforces_balanced_answer_type_quotas(tmp_path):
    module = _module()
    image_root = tmp_path / "images"
    image_root.mkdir()
    for index in range(5):
        Image.new("RGB", (6, 6), color="pink").save(image_root / f"case-{index}.png")
    records = [
        {
            "image_name": f"case-{index}.png",
            "question": f"Question {index}?",
            "answer": "yes" if index < 3 else f"finding {index}",
        }
        for index in range(5)
    ]
    source = tmp_path / "pathvqa.json"
    source.write_text(json.dumps(records), encoding="utf-8")
    output = tmp_path / "normalized"
    result = module.export_medical_vqa_dataset(
        dataset="pathvqa",
        revision="1685832883334b5bb5beaf4e4b333fdeecaa4ad9",
        split="test",
        output_directory=output,
        closed_samples=2,
        open_samples=2,
        source_path=source,
        image_root=image_root,
    )
    assert result["sample_count"] == 4
    assert result["answer_type_counts"] == {"closed": 2, "open": 2}
