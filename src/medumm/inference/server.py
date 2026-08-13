from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from medumm.core.io import ensure_directory, write_json
from medumm.core.runtime import redact_secrets
from medumm.inference.backends import (
    BackendConfig,
    BackendMode,
    InferenceBackend,
    backend_catalog,
)


_MUTABLE_REVISIONS = frozenset({"", "main", "master", "latest", "head"})


def _flag_tokens(values: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    for key, value in values.items():
        flag = "--" + str(key).replace("_", "-")
        if value is None or value is False:
            continue
        if value is True:
            tokens.append(flag)
        elif isinstance(value, list):
            for item in value:
                tokens.extend([flag, str(item)])
        else:
            tokens.extend([flag, str(value)])
    return tokens


def server_command(config: dict[str, Any]) -> tuple[list[str], BackendConfig]:
    backend = BackendConfig.from_dict(config.get("backend"))
    if backend.name is InferenceBackend.NATIVE:
        raise ValueError("Inference serving requires backend.name=vllm or sglang.")
    if backend.mode is not BackendMode.OPENAI_HTTP:
        raise ValueError("Inference serving requires backend.mode=openai_http.")
    model_path = str(config.get("model_path", "")).strip()
    revision = str(config.get("model_revision", "")).strip()
    if not model_path:
        raise ValueError("server.model_path is required.")
    if revision.casefold() in _MUTABLE_REVISIONS:
        raise ValueError("server.model_revision must be immutable.")
    host = str(config.get("host", "127.0.0.1"))
    port = int(config.get("port", 8000))
    if not 1 <= port <= 65535:
        raise ValueError("server.port must be between 1 and 65535.")
    served_name = str(config.get("served_model_name", model_path))
    parallel = backend.parallel
    scheduler = backend.scheduler
    common = {
        "host": host,
        "port": port,
        "trust_remote_code": bool(config.get("trust_remote_code", False)),
    }
    if backend.name is InferenceBackend.VLLM:
        command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
        arguments = {
            "model": model_path,
            "revision": revision,
            "served_model_name": served_name,
            "tensor_parallel_size": parallel.tensor_parallel_size,
            "pipeline_parallel_size": parallel.pipeline_parallel_size,
            "data_parallel_size": parallel.data_parallel_size,
            "distributed_executor_backend": parallel.distributed_executor_backend,
            "max_num_seqs": scheduler.max_num_seqs,
            "max_num_batched_tokens": scheduler.max_num_batched_tokens,
            "gpu_memory_utilization": float(config.get("gpu_memory_utilization", 0.9)),
            **common,
        }
    else:
        command = [sys.executable, "-m", "sglang.launch_server"]
        arguments = {
            "model_path": model_path,
            "revision": revision,
            "served_model_name": served_name,
            "tp_size": parallel.tensor_parallel_size,
            "pp_size": parallel.pipeline_parallel_size,
            "dp_size": parallel.data_parallel_size,
            "max_running_requests": scheduler.max_num_seqs,
            "max_prefill_tokens": scheduler.max_num_batched_tokens,
            "max_queued_requests": scheduler.max_queue_size,
            "mem_fraction_static": float(config.get("gpu_memory_utilization", 0.9)),
            **common,
        }
    extra = config.get("extra_args", {}) or {}
    if not isinstance(extra, dict):
        raise ValueError("server.extra_args must be a mapping.")
    return command + _flag_tokens({**arguments, **extra}), backend


def plan_server(
    config: dict[str, Any], *, project_root: Path, require_installed: bool = False
) -> dict[str, Any]:
    command, backend = server_command(config)
    output = Path(str(config.get("output_directory", "outputs/serving")))
    if not output.is_absolute():
        output = project_root / output
    ensure_directory(output)
    capabilities = {item["name"]: item for item in backend_catalog()}[backend.name.value]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_count = (
        len([item for item in visible.split(",") if item.strip()]) if visible else None
    )
    errors: list[str] = []
    warnings: list[str] = []
    if require_installed and not capabilities["installed"]:
        errors.append(f"backend {backend.name.value} is not installed")
    elif not capabilities["installed"]:
        warnings.append(f"backend {backend.name.value} is not installed in this plan runtime")
    if visible_count is not None and backend.parallel.world_size > visible_count:
        errors.append(
            f"parallel world size {backend.parallel.world_size} exceeds "
            f"CUDA_VISIBLE_DEVICES count {visible_count}"
        )
    plan = {
        "schema_version": "1.0",
        "status": "blocked" if errors else "ready",
        "backend": backend.to_dict(),
        "capabilities": capabilities,
        "command": command,
        "command_text": shlex.join(command),
        "configuration": redact_secrets(config),
        "visible_gpu_count": visible_count,
        "errors": errors,
        "warnings": warnings,
        "output_directory": str(output),
    }
    write_json(output / "server_plan.json", plan)
    return plan


def launch_server(config: dict[str, Any], *, project_root: Path) -> int:
    plan = plan_server(config, project_root=project_root, require_installed=True)
    if plan["status"] != "ready":
        raise RuntimeError("Inference server preflight failed: " + "; ".join(plan["errors"]))
    output = Path(plan["output_directory"])
    log_path = output / "server.log"
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            plan["command"],
            cwd=project_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(completed.returncode)
