from medumm import EvaluationProtocol, MetricSuite, catalog, create_metric_suite, infer
from tests.conftest import PROJECT_ROOT


def test_catalog_exposes_four_plugin_kinds():
    components = catalog()
    assert set(components) == {"models", "datasets", "benchmarks", "trainers"}
    assert {item["name"] for item in components["benchmarks"]} >= {
        "medical_vqa",
        "cross_task",
    }
    llava_med = next(item for item in components["models"] if item["name"] == "llava_med")
    assert llava_med["metadata"]["default_model"] == (
        "microsoft/llava-med-v1.5-mistral-7b"
    )


def test_high_level_inference_api_accepts_unified_config():
    results = infer(
        {
            "schema_version": "1.0",
            "inference": {
                "backbone": "medical_reference",
                "request": {
                    "id": "api-test",
                    "task": "understanding",
                    "prompt": "Answer A.",
                    "parameters": {"fixed_answer": "A"},
                },
            },
        },
        config_path=PROJECT_ROOT / "pyproject.toml",
    )
    assert results[0].request_id == "api-test"
    assert results[0].text == "A"


def test_evaluation_protocol_and_metric_suite_are_public():
    protocol = EvaluationProtocol()
    suite = create_metric_suite(protocol.metric_suite)
    assert isinstance(suite, MetricSuite)
    assert suite.version == protocol.metric_suite_version
