from medumm.medical import load_medical_vqa
from tests.conftest import PROJECT_ROOT


def test_loads_medical_vqa_schema():
    samples = load_medical_vqa(
        {
            "path": "examples/medical/tiny_eval.jsonl",
            "image_root": "examples/medical/images",
        },
        project_root=PROJECT_ROOT,
    )
    assert len(samples) == 3
    assert samples[0].choices == {"A": "yes", "B": "no"}
    assert samples[0].image_paths[0].endswith("synthetic_scan.pgm")
