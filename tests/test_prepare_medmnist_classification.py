from importlib.util import module_from_spec, spec_from_file_location
import hashlib
import json

import numpy as np
import pytest

from tests.conftest import PROJECT_ROOT


def _module():
    path = PROJECT_ROOT / "scripts/prepare_medmnist_classification.py"
    spec = spec_from_file_location("prepare_medmnist_classification", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _archive(path):
    np.savez(
        path,
        test_images=np.array([[[0, 64], [128, 255]], [[255, 128], [64, 0]]], dtype=np.uint8),
        test_labels=np.array([[0], [1]], dtype=np.uint8),
    )


def test_pneumoniamnist_export_preserves_medical_labels_and_choices(tmp_path, monkeypatch):
    module = _module()
    source = tmp_path / "pneumoniamnist.npz"
    output = tmp_path / "normalized"
    _archive(source)
    monkeypatch.setitem(
        module.DATASETS["pneumoniamnist"],
        "md5",
        hashlib.md5(source.read_bytes()).hexdigest(),
    )
    result = module.export_medmnist_classification(
        dataset="pneumoniamnist",
        revision="v2",
        split="test",
        npz_path=source,
        output_directory=output,
    )
    rows = [json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()]
    assert result["sample_count"] == 2
    assert rows[0]["answer"] == "normal chest x-ray"
    assert rows[1]["answer"] == "pneumonia chest x-ray"
    assert rows[0]["choices"] == {
        "A": "normal chest x-ray",
        "B": "pneumonia chest x-ray",
    }
    assert rows[0]["modality"] == "chest_xray"
    assert (output / "images" / "smoke.png").is_file()


def test_pneumoniamnist_export_rejects_wrong_release(tmp_path):
    module = _module()
    with pytest.raises(ValueError, match="fixed v2"):
        module.export_medmnist_classification(
            dataset="pneumoniamnist",
            revision="main",
            split="test",
            npz_path=tmp_path / "missing.npz",
            output_directory=tmp_path / "output",
        )
