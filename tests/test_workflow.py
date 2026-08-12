import json

from medumm.evaluation import run_medical_vqa
from medumm.inference import InferencePipeline
from medumm.post_training import PostTrainingRunner
from tests.conftest import PROJECT_ROOT


def test_train_infer_evaluate_workflow(tmp_path):
    training = {
        "method": "medical_sft",
        "data": {
            "path": "examples/medical/tiny_train.jsonl",
            "image_root": "examples/medical/images",
            "deidentified": True,
        },
        "text_dimensions": 64,
        "epochs": 80,
        "output_directory": str(tmp_path / "model"),
    }
    trained = PostTrainingRunner().run(training, config_path=PROJECT_ROOT / "pyproject.toml")
    assert trained["status"] == "completed"
    assert trained["train_accuracy"] >= 0.99

    with InferencePipeline("medical_linear", {"model_path": str(tmp_path / "model")}) as pipeline:
        output = pipeline.run({
            "task": "understanding",
            "prompt": "The synthetic centre is brighter than the border. Choose A for yes or B for no.",
            "images": [str(PROJECT_ROOT / "examples/medical/images/synthetic_scan.pgm")],
        })
    assert output["understandings"][0]["response"] == "A"

    evaluation = {
        "benchmark": "medical_vqa",
        "data": {
            "path": "examples/medical/tiny_eval.jsonl",
            "image_root": "examples/medical/images",
        },
        "model": {"backbone": "medical_linear", "config": {"model_path": str(tmp_path / "model")}},
        "evaluation": {"output_directory": str(tmp_path / "evaluation"), "resume": False},
    }
    report = run_medical_vqa(evaluation, config_path=PROJECT_ROOT / "pyproject.toml")
    score = json.loads((tmp_path / "evaluation/score.json").read_text())
    assert report["status"] == "completed"
    assert score["metrics"]["overall"]["exact_match"] >= 66.0
