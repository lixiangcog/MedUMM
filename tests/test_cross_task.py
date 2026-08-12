from medumm.core.runtime import RuntimeContext
from medumm.evaluation import CrossTaskBenchmark
from tests.conftest import PROJECT_ROOT


def test_cross_task_benchmark_composes_registered_benchmarks(tmp_path):
    runtime = RuntimeContext.create(
        command="evaluation",
        config_path=PROJECT_ROOT / "pyproject.toml",
        output_directory=tmp_path / "cross",
    )
    benchmark = CrossTaskBenchmark()
    result = benchmark.run(
        {
            "output_directory": str(tmp_path / "cross"),
            "benchmarks": [
                {
                    "name": "understanding",
                    "benchmark": "medical_vqa",
                    "data": {
                        "path": "examples/medical/tiny_eval.jsonl",
                        "image_root": "examples/medical/images",
                    },
                    "model": {
                        "backbone": "medical_reference",
                        "parameters": {"fixed_answer": "A"},
                    },
                    "mode": "full",
                    "resume": False,
                }
            ],
        },
        config_path=PROJECT_ROOT / "pyproject.toml",
        runtime=runtime,
    )
    assert result.status == "completed"
    assert result.metrics["summary"]["benchmark_count"] == 1
    assert (tmp_path / "cross/cross_task_report.json").is_file()
