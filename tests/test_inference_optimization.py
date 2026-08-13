from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from medumm.backbones.emu3_5 import _DEFAULT_SAMPLING, Emu3_5Adapter
from medumm.cli.main import main
from medumm.core.builtins import register_builtins
from medumm.core.contracts import TaskType
from medumm.core.registry import registry
from medumm.core.results import InferenceResult
from medumm.core.runtime import RuntimeContext
from medumm.inference import BackendConfig, backend_catalog
from medumm.inference.benchmark import run_inference_benchmark
from medumm.inference.openai_backend import OpenAIHTTPAdapter
from medumm.inference.request import InferenceRequest
from medumm.inference.server import plan_server, server_command
from tests.conftest import PROJECT_ROOT


def _runtime(tmp_path: Path) -> RuntimeContext:
    return RuntimeContext.create(
        command="test",
        config_path=PROJECT_ROOT / "pyproject.toml",
        output_directory=tmp_path,
    )


def test_backend_catalog_exposes_real_capability_boundaries():
    values = {item["name"]: item for item in backend_catalog()}
    assert set(values) == {"native", "vllm", "sglang"}
    assert values["vllm"]["continuous_batching"] is True
    assert values["vllm"]["tensor_parallel"] is True
    assert values["sglang"]["emu3_5_native_cfg"] is False
    assert values["native"]["continuous_batching"] is False


def test_backend_config_validates_parallel_world_size():
    config = BackendConfig.from_dict(
        {
            "name": "vllm",
            "mode": "openai_http",
            "endpoint": "http://127.0.0.1:8000",
            "parallel": {
                "tensor_parallel_size": 2,
                "pipeline_parallel_size": 2,
                "data_parallel_size": 2,
            },
        }
    )
    assert config.parallel.world_size == 8
    assert config.continuous_batching is True


@pytest.mark.parametrize(
    "value,match",
    [
        ({"name": "native", "parallel": {"tensor_parallel_size": 2}}, "model parallelism"),
        ({"name": "sglang", "mode": "in_process"}, "OpenAI HTTP"),
        ({"name": "vllm", "mode": "openai_http"}, "endpoint"),
    ],
)
def test_backend_config_rejects_unsupported_combinations(value, match):
    with pytest.raises(ValueError, match=match):
        BackendConfig.from_dict(value)


def test_registered_emu3_5_capabilities_declare_optimization_contract():
    register_builtins()
    adapter = registry.models.create("emu3_5")
    capabilities = adapter.capabilities.to_dict()
    assert capabilities["supported_backends"] == ["vllm"]
    assert capabilities["supports_continuous_batching"] is True
    assert capabilities["supports_classifier_free_guidance"] is True
    assert capabilities["parallelism"] == ["tensor_parallel"]


def test_emu3_5_refuses_sglang_cfg_before_model_loading(tmp_path):
    with pytest.raises(RuntimeError, match="does not provide.*CFG scheduler"):
        Emu3_5Adapter().load(
            {
                "backend": {
                    "name": "sglang",
                    "mode": "openai_http",
                    "endpoint": "http://127.0.0.1:30000",
                }
            },
            _runtime(tmp_path),
        )


def test_emu3_5_refuses_unpatched_missing_vllm(tmp_path, monkeypatch):
    def missing(_config):
        raise RuntimeError("Inference backend 'vllm' is not installed in this runtime.")

    monkeypatch.setattr("medumm.backbones.emu3_5.require_backend", missing)
    with pytest.raises(RuntimeError, match="not installed"):
        Emu3_5Adapter().load(
            {"backend": {"name": "vllm", "mode": "in_process"}},
            _runtime(tmp_path),
        )


def test_emu3_5_refuses_unpatched_vllm_before_model_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "medumm.backbones.emu3_5.require_backend",
        lambda _config: {
            "installed": True,
            "version": "0.11.0",
            "emu3_5_native_cfg": False,
        },
    )
    with pytest.raises(RuntimeError, match="complete vLLM patch set"):
        Emu3_5Adapter().load(
            {
                "backend": {"name": "vllm", "mode": "in_process"},
                "source_revision": "0123456789abcdef",
                "model_revision": "0123456789abcdef",
                "vq_revision": "0123456789abcdef",
            },
            _runtime(tmp_path),
        )


def test_vllm_server_command_contains_batch_and_parallel_controls():
    command, backend = server_command(
        {
            "model_path": "/models/pinned",
            "model_revision": "0123456789abcdef",
            "backend": {
                "name": "vllm",
                "mode": "openai_http",
                "endpoint": "http://127.0.0.1:8000",
                "parallel": {
                    "tensor_parallel_size": 2,
                    "pipeline_parallel_size": 2,
                    "data_parallel_size": 1,
                },
                "scheduler": {
                    "continuous_batching": True,
                    "max_num_seqs": 16,
                    "max_num_batched_tokens": 4096,
                },
            },
        }
    )
    text = " ".join(command)
    assert backend.parallel.world_size == 4
    assert "--tensor-parallel-size 2" in text
    assert "--pipeline-parallel-size 2" in text
    assert "--max-num-seqs 16" in text
    assert "--max-num-batched-tokens 4096" in text


def test_sglang_server_command_maps_parallel_controls():
    command, _ = server_command(
        {
            "model_path": "/models/pinned",
            "model_revision": "0123456789abcdef",
            "backend": {
                "name": "sglang",
                "mode": "openai_http",
                "endpoint": "http://127.0.0.1:30000",
                "parallel": {
                    "tensor_parallel_size": 2,
                    "pipeline_parallel_size": 2,
                    "data_parallel_size": 2,
                },
                "scheduler": {"max_num_seqs": 24},
            },
        }
    )
    text = " ".join(command)
    assert "--tp-size 2" in text
    assert "--pp-size 2" in text
    assert "--dp-size 2" in text
    assert "--max-running-requests 24" in text
    assert "--max-prefill-tokens 8192" in text
    assert "--max-queued-requests 1024" in text


def test_server_plan_records_unavailable_runtime_without_claiming_execution(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "medumm.inference.server.backend_catalog",
        lambda: [{"name": "vllm", "installed": False}],
    )
    plan = plan_server(
        {
            "model_path": "/models/pinned",
            "model_revision": "0123456789abcdef",
            "output_directory": str(tmp_path / "plan"),
            "backend": {
                "name": "vllm",
                "mode": "openai_http",
                "endpoint": "http://127.0.0.1:8000",
            },
        },
        project_root=PROJECT_ROOT,
    )
    assert plan["status"] == "ready"
    assert plan["capabilities"]["installed"] is False
    assert plan["warnings"]
    assert (tmp_path / "plan/server_plan.json").is_file()


def test_http_adapter_sends_batch_concurrently_and_preserves_order(tmp_path, monkeypatch):
    adapter = OpenAIHTTPAdapter()
    adapter.load(
        {
            "model": "medical-test-model",
            "backend": {
                "name": "sglang",
                "mode": "openai_http",
                "endpoint": "http://127.0.0.1:30000",
                "scheduler": {"continuous_batching": True, "max_num_seqs": 4},
            },
        },
        _runtime(tmp_path),
    )

    def fake(request):
        if request.request_id == "slow":
            time.sleep(0.02)
        return InferenceResult(request.request_id, "understanding", "openai_http", text="ok")

    monkeypatch.setattr(adapter, "_request", fake)
    requests = [
        InferenceRequest(task="understanding", request_id="slow", prompt="one"),
        InferenceRequest(task="understanding", request_id="fast", prompt="two"),
    ]
    assert [item.request_id for item in adapter.understand_batch(requests)] == ["slow", "fast"]


def test_http_adapter_rejects_emu_cfg_controls(tmp_path):
    adapter = OpenAIHTTPAdapter()
    adapter.load(
        {
            "model": "medical-test-model",
            "backend": {
                "name": "vllm",
                "mode": "openai_http",
                "endpoint": "http://127.0.0.1:8000",
            },
        },
        _runtime(tmp_path),
    )
    request = InferenceRequest(
        task="understanding",
        prompt="test",
        parameters={"classifier_free_guidance": 5.0},
    )
    with pytest.raises(ValueError, match="does not expose Emu3.5 CFG"):
        adapter._payload(request)


def test_http_adapter_uses_portable_text_only_message(tmp_path):
    adapter = OpenAIHTTPAdapter()
    adapter.load(
        {
            "model": "medical-test-model",
            "backend": {
                "name": "vllm",
                "mode": "openai_http",
                "endpoint": "http://127.0.0.1:8000",
            },
        },
        _runtime(tmp_path),
    )
    payload = adapter._payload(
        InferenceRequest(task="understanding", prompt="portable text request")
    )
    assert payload["messages"][-1]["content"] == "portable text request"


def test_emu_batch_submits_all_cond_uncond_pairs_in_one_engine_call(monkeypatch):
    adapter = Emu3_5Adapter()
    captured = {}

    class Engine:
        def generate(self, inputs, sampling_params, use_tqdm):
            captured["inputs"] = inputs
            captured["sampling"] = sampling_params
            return [SimpleNamespace(request_id=str(index)) for index in range(len(inputs))]

    adapter.model = Engine()
    adapter.backend = BackendConfig.from_dict({"name": "vllm", "mode": "in_process"})
    monkeypatch.setattr(
        adapter,
        "_prepare",
        lambda request: (
            {"prompt_token_ids": [1], "uncond_prompt_token_ids": [2]},
            {"temperature": 1},
            "t2i",
        ),
    )
    monkeypatch.setattr(
        adapter,
        "_decode",
        lambda request, output, task: InferenceResult(
            request.request_id, request.task, "emu3_5", metadata={}
        ),
    )
    requests = [
        InferenceRequest(task="generation", request_id="a", prompt="a"),
        InferenceRequest(task="generation", request_id="b", prompt="b"),
    ]
    results = adapter._run_batch(requests)
    assert [result.request_id for result in results] == ["a", "b"]
    assert len(captured["inputs"]) == 2
    assert all("uncond_prompt_token_ids" in item for item in captured["inputs"])
    assert len(captured["sampling"]) == 2


def test_emu_defaults_include_all_sampling_controls():
    adapter = Emu3_5Adapter()
    configured = {}
    adapter.defaults = {
        **_DEFAULT_SAMPLING,
        "classifier_free_guidance": 5.0,
        "image_area": 1048576,
        **configured,
    }
    for key in (
        "text_top_k",
        "text_top_p",
        "text_temperature",
        "image_top_k",
        "image_top_p",
        "image_temperature",
        "max_new_tokens",
    ):
        assert key in adapter.defaults


def test_reference_performance_benchmark_writes_reproducible_metrics(tmp_path):
    config = {
        "schema_version": "1.0",
        "inference": {
            "backbone": "medical_reference",
            "config": {},
            "requests": [
                {
                    "request_id": "a",
                    "task": "understanding",
                    "prompt": "synthetic",
                    "parameters": {"fixed_answer": "A"},
                },
                {
                    "request_id": "b",
                    "task": "understanding",
                    "prompt": "synthetic",
                    "parameters": {"fixed_answer": "B"},
                },
            ],
        },
        "benchmark": {
            "warmup_iterations": 1,
            "measured_iterations": 2,
            "batch_size": 2,
            "output_directory": str(tmp_path / "benchmark"),
        },
    }
    report = run_inference_benchmark(config, config_path=PROJECT_ROOT / "pyproject.toml")
    stored = json.loads((tmp_path / "benchmark/benchmark.json").read_text())
    assert report["status"] == "completed"
    assert report["benchmark"]["total_requests"] == 4
    assert report["throughput"]["requests_per_second"] > 0
    assert stored["runtime"]["backbone"] == "medical_reference"
    assert stored["clinical_use"] is False


def test_backend_and_benchmark_cli(capsys, tmp_path):
    assert main(["backends", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 3
    assert main(
        [
            "benchmark-inference",
            "--config",
            str(PROJECT_ROOT / "configs/inference/benchmark_reference_v1.2.yaml"),
            "--set",
            f"benchmark.output_directory={tmp_path / 'cli-benchmark'}",
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["benchmark"]["total_requests"] == 20
