import json

import pytest

from medumm.core import EvaluationMode, InferenceResult, TaskType
from medumm.evaluation.runner import EvaluationItem, EvaluationRunner


def _items():
    return [EvaluationItem("one", {}, {"answer": "A"})]


def _runner(tmp_path, mode, *, resume=False):
    return EvaluationRunner(
        benchmark="test",
        pipeline=None,
        output_directory=tmp_path,
        parser=lambda result: str(result.text),
        scorer=lambda prediction, content: {
            "exact": float(prediction == content["answer"])
        },
        summarizer=lambda rows: {"overall": {"total": len(rows), "exact": rows[0]["exact"]}},
        mode=mode,
        resume=resume,
        fingerprint="stable",
    )


def test_score_mode_reads_existing_predictions_even_when_resume_is_false(tmp_path):
    prediction = {
        "id": "one",
        "request_id": "one",
        "prediction": "A",
        "fingerprint": "stable",
        "model_name": "test",
    }
    (tmp_path / "predictions.jsonl").write_text(json.dumps(prediction) + "\n")
    result = _runner(tmp_path, EvaluationMode.SCORE).run(_items())
    assert result.status == "completed"
    assert result.metrics["overall"]["exact"] == 1.0


def test_score_mode_requires_predictions(tmp_path):
    with pytest.raises(FileNotFoundError, match="requires predictions"):
        _runner(tmp_path, EvaluationMode.SCORE).run(_items())
