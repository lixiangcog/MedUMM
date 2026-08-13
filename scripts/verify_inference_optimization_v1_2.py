from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as reader:
        value = json.load(reader)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--sequential", required=True)
    parser.add_argument("--concurrent", required=True)
    parser.add_argument("--server-plan", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    sequential = _read(arguments.sequential)
    concurrent = _read(arguments.concurrent)
    server = _read(arguments.server_plan)
    for label, report in (("sequential", sequential), ("concurrent", concurrent)):
        if report.get("status") != "completed":
            raise RuntimeError(f"{label} benchmark did not complete")
        runtime_backend = report["runtime"]["runtime"]["backend"]
        if runtime_backend["name"] != arguments.backend:
            raise RuntimeError(f"{label} benchmark used the wrong backend")
        if runtime_backend["parallel"]["world_size"] != 2:
            raise RuntimeError(f"{label} benchmark did not record two-GPU parallelism")
        if report["benchmark"]["total_requests"] != 24:
            raise RuntimeError(f"{label} benchmark did not execute all requests")
    if sequential["benchmark"]["batch_size"] != 1:
        raise RuntimeError("Sequential benchmark must use batch_size=1")
    if concurrent["benchmark"]["batch_size"] != 8:
        raise RuntimeError("Concurrent benchmark must use batch_size=8")
    if not concurrent["runtime"]["runtime"]["backend"]["continuous_batching"]:
        raise RuntimeError("Continuous batching was not enabled")
    if server.get("status") != "ready":
        raise RuntimeError("Server preflight did not pass")

    sequential_rate = float(sequential["throughput"]["requests_per_second"])
    concurrent_rate = float(concurrent["throughput"]["requests_per_second"])
    result = {
        "schema_version": "1.0",
        "status": "passed",
        "backend": arguments.backend,
        "parallel_world_size": 2,
        "tensor_parallel_size": 2,
        "continuous_batching": True,
        "requests_per_profile": 24,
        "sequential_requests_per_second": sequential_rate,
        "concurrent_requests_per_second": concurrent_rate,
        "concurrent_throughput_ratio": round(concurrent_rate / sequential_rate, 6),
        "sequential_report": str(Path(arguments.sequential).resolve()),
        "concurrent_report": str(Path(arguments.concurrent).resolve()),
        "server_plan": str(Path(arguments.server_plan).resolve()),
        "clinical_use": False,
    }
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
