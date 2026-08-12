from medumm.medical.dataset import MedicalVQADatasetAdapter
from tests.conftest import PROJECT_ROOT


def test_dataset_fingerprint_changes_with_sample_limit():
    adapter = MedicalVQADatasetAdapter()
    base = {"path": "examples/medical/tiny_eval.jsonl", "image_root": "examples/medical/images"}
    limited = {**base, "max_samples": 1}
    assert adapter.fingerprint(base, PROJECT_ROOT) != adapter.fingerprint(limited, PROJECT_ROOT)
