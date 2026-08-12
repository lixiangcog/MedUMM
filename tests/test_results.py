from medumm.core import EvaluationResult, InferenceResult, TaskType, TrainingResult


def test_all_public_results_have_a_schema_version():
    inference = InferenceResult("one", "understanding", "reference")
    training = TrainingResult("sft", "completed", "/tmp/output")
    evaluation = EvaluationResult("medical_vqa", "full", "completed", 2, "/tmp/output")
    assert inference.task is TaskType.UNDERSTANDING
    assert inference.to_dict()["schema_version"] == "1.0"
    assert training.to_dict()["schema_version"] == "1.0"
    assert evaluation.to_dict()["schema_version"] == "1.0"
