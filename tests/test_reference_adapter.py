from pathlib import Path

from medumm.inference import InferencePipeline
from tests.conftest import PROJECT_ROOT


def test_reference_adapter_supports_three_tasks(tmp_path):
    image = PROJECT_ROOT / "examples/medical/images/synthetic_scan.pgm"
    with InferencePipeline("medical_reference", {"output_directory": str(tmp_path)}) as pipeline:
        generated = pipeline.run({"task": "generation", "prompt": "phantom"})
        understood = pipeline.run({
            "task": "understanding",
            "prompt": "bright?",
            "images": [str(image)],
            "parameters": {"fixed_answer": "A"},
        })
        edited = pipeline.run({
            "task": "editing",
            "prompt": "increase contrast",
            "images": [str(image)],
        })
    assert Path(generated.output_path).is_file()
    assert understood.text == "A"
    assert Path(edited.output_path).is_file()
    assert generated.model_name == "medical_reference"
    assert pipeline.capabilities.supports("editing")
