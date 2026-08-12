from pathlib import Path

import pytest

from medumm.core.exceptions import UnsupportedTaskError
from medumm.inference import InferencePipeline
from medumm.inference import InferenceRequest
from tests.conftest import PROJECT_ROOT


def test_mixed_requests_are_batched_by_task_and_returned_in_order(tmp_path):
    image = PROJECT_ROOT / "examples/medical/images/synthetic_scan.pgm"
    with InferencePipeline("medical_reference", {"output_directory": str(tmp_path)}) as pipeline:
        results = pipeline.run_many(
            [
                {"id": "u", "task": "understanding", "prompt": "bright?", "images": [str(image)]},
                {"id": "g", "task": "generation", "prompt": "phantom"},
                {"id": "e", "task": "editing", "prompt": "increase contrast", "images": [str(image)]},
            ],
            batch_size=2,
        )
    assert [result.request_id for result in results] == ["u", "g", "e"]
    assert Path(results[1].output_path).is_file()


def test_model_load_validates_required_checkpoint(tmp_path):
    model = tmp_path / "missing-model"
    with pytest.raises(FileNotFoundError):
        InferencePipeline("medical_linear", {"model_path": str(model)})


def test_pipeline_rejects_request_for_another_model():
    with InferencePipeline("medical_reference", {}) as pipeline:
        with pytest.raises(ValueError, match=r"request\(s\) specify"):
            pipeline.run({
                "model": "medgemma",
                "task": "generation",
                "prompt": "synthetic image",
            })


def test_pipeline_does_not_mutate_request_paths():
    request = InferenceRequest(
        task="understanding",
        prompt="Describe",
        images=["examples/medical/images/synthetic_scan.pgm"],
    )
    with InferencePipeline("medical_reference", {}) as pipeline:
        pipeline.run(request)
    assert request.images == ["examples/medical/images/synthetic_scan.pgm"]


def test_pipeline_rejects_an_empty_request_batch():
    with InferencePipeline("medical_reference", {}) as pipeline:
        with pytest.raises(ValueError, match="at least one request"):
            pipeline.run_many([])
