from importlib.util import module_from_spec, spec_from_file_location
import json

from PIL import Image

from tests.conftest import PROJECT_ROOT


def _module():
    path = PROJECT_ROOT / "scripts/prepare_vqa_rad.py"
    spec = spec_from_file_location("prepare_vqa_rad", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_vqa_rad_export_uses_stable_identifiers():
    module = _module()
    assert module._record_id("test", 3, "Is this normal?") == module._record_id(
        "test", 3, "Is this normal?"
    )
    assert module._record_id("test", 3, "Is this normal?").startswith("vqa-rad-test-00003-")


def test_vqa_rad_script_parser_accepts_closed_only(tmp_path):
    module = _module()
    values = module.build_parser().parse_args(
        ["--output-directory", str(tmp_path), "--closed-only", "--max-samples", "2"]
    )
    assert values.closed_only
    assert values.max_samples == 2


def test_vqa_rad_export_accepts_official_json_and_images(tmp_path):
    module = _module()
    source = tmp_path / "official.json"
    image_root = tmp_path / "official-images"
    output = tmp_path / "normalized"
    image_root.mkdir()
    Image.new("RGB", (4, 4), color="white").save(image_root / "case.jpg")
    source.write_text(
        json.dumps(
            [
                {
                    "qid": 7,
                    "image_name": "case.jpg",
                    "question": "Is this normal?",
                    "answer": "No",
                    "answer_type": "CLOSED",
                    "question_type": "ABN",
                }
            ]
        ),
        encoding="utf-8",
    )

    provenance = module.export_vqa_rad(
        dataset_name="VQA-RAD/official",
        revision="osf-file-version-1",
        split="acceptance",
        output_directory=output,
        max_samples=1,
        closed_only=True,
        json_path=source,
        image_root=image_root,
        source_mirror="https://osf.io/89kps/",
    )

    record = json.loads((output / "samples.jsonl").read_text())
    assert provenance["resolved_revision"] == "osf-file-version-1"
    assert provenance["source_json"] == str(source)
    assert record["metadata"]["source_qid"] == 7
    assert len(record["metadata"]["source_image_sha256"]) == 64
