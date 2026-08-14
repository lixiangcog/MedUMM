#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


EXPECTED = {
    "medmo_4b": {
        "revision": "0e220705d851598b37725326aadb852aa8b37f43",
        "executor": "qwen3_vl_chat",
        "transformers": "4.57.1",
    },
    "medmo_8b": {
        "revision": "8eafab80545fb4d60b0fb126a097e972e6475851",
        "executor": "qwen3_vl_chat",
        "transformers": "4.57.1",
    },
    "lingshu_i_8b": {
        "revision": "b004bfc0554d90bd44baedf4de08c361e71ef017",
        "executor": "internvl_transformers",
        "transformers": "4.52.4",
    },
    "fleming_vl_8b": {
        "revision": "801e2bef9645bca0646d55837a6630fb468e2901",
        "executor": "internvl_chat",
        "transformers": "4.46.0",
    },
}


def _read_result(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise AssertionError(f"Expected one inference result in {path}.")
    return value[0]


def _environment(environment_root: Path, model: str) -> dict[str, Any]:
    directory = environment_root / model
    python = directory / "bin/python"
    fingerprint = directory / "medumm-environment.sha256"
    if not python.is_file() or not fingerprint.is_file():
        raise FileNotFoundError(f"Missing isolated environment evidence for {model}: {directory}")
    program = (
        "import json,platform,torch,transformers,medumm; "
        "print(json.dumps({'python': platform.python_version(), 'medumm': medumm.__version__, "
        "'torch': torch.__version__, 'transformers': transformers.__version__, "
        "'cuda': torch.version.cuda}))"
    )
    process = subprocess.run(
        [str(python), "-c", program],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "path": str(directory.resolve()),
        "contract_sha256": fingerprint.read_text(encoding="utf-8").strip(),
        **json.loads(process.stdout),
    }


def verify(
    *,
    results: dict[str, Path],
    provenance_path: Path,
    environment_root: Path,
    output: Path,
) -> dict[str, Any]:
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if not slurm_job_id:
        raise AssertionError("Runtime acceptance must execute inside a Slurm allocation.")
    rows = []
    for model, path in results.items():
        expected = EXPECTED[model]
        result = _read_result(path)
        metadata = dict(result.get("metadata", {}))
        if result.get("model_name") != model:
            raise AssertionError(f"{model} returned the wrong model identity.")
        if metadata.get("model_revision") != expected["revision"]:
            raise AssertionError(f"{model} did not report its pinned model revision.")
        if metadata.get("executor") != expected["executor"]:
            raise AssertionError(f"{model} did not use the expected explicit executor.")
        if not str(metadata.get("device", "")).startswith("cuda"):
            raise AssertionError(f"{model} did not execute on CUDA.")
        if metadata.get("scheduler", {}).get("slurm_job_id") != slurm_job_id:
            raise AssertionError(f"{model} has no matching Slurm provenance.")
        if not str(result.get("text", "")).strip():
            raise AssertionError(f"{model} produced an empty response.")
        if result.get("duration_ms") is None or float(result["duration_ms"]) <= 0:
            raise AssertionError(f"{model} has no measured inference latency.")
        asset = provenance.get("models", {}).get(model, {})
        if asset.get("revision") != expected["revision"]:
            raise AssertionError(f"{model} asset provenance does not match the recipe.")
        environment = _environment(environment_root, model)
        if environment.get("medumm") != "1.5.0":
            raise AssertionError(f"{model} did not run the MedUMM 1.5.0 package.")
        if environment.get("transformers") != expected["transformers"]:
            raise AssertionError(f"{model} used an unexpected Transformers release.")
        rows.append(
            {
                "model": model,
                "model_revision": expected["revision"],
                "executor": expected["executor"],
                "device": metadata["device"],
                "dtype": metadata.get("dtype"),
                "duration_ms": result["duration_ms"],
                "peak_gpu_memory_mb": metadata.get("peak_gpu_memory_mb"),
                "response": result["text"],
                "result_path": str(path),
                "environment": environment,
            }
        )
    try:
        import torch

        runtime = {
            "torch": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ModuleNotFoundError:
        runtime = {"torch": None, "cuda_available": False, "gpu": None}
    if not runtime["cuda_available"]:
        raise AssertionError("The verification process cannot see the allocated GPU.")
    evidence = {
        "schema_version": "1.0",
        "release": "v1.5.0",
        "status": "passed",
        "hostname": platform.node(),
        "scheduler": {"slurm_job_id": slurm_job_id},
        "runtime": runtime,
        "validated_models": rows,
        "gated_access_probe": provenance.get("gated_access_probe", {}),
        "counts": {
            "validated_in_this_job": len(rows),
            "runtime_validated_after_release": 11,
            "catalog_models": 32,
        },
        "validation_scope": (
            "Pinned real-weight inference through the public MedUMM adapter interface on one "
            "Slurm-allocated GPU. This is interface/runtime evidence, not a clinical-quality claim."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify MedUMM v1.5 real model adapters")
    parser.add_argument("--medmo-4b", type=Path, required=True)
    parser.add_argument("--medmo-8b", type=Path, required=True)
    parser.add_argument("--lingshu-i-8b", type=Path, required=True)
    parser.add_argument("--fleming-vl-8b", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(arguments: list[str] | None = None) -> int:
    values = build_parser().parse_args(arguments)
    evidence = verify(
        results={
            "medmo_4b": values.medmo_4b,
            "medmo_8b": values.medmo_8b,
            "lingshu_i_8b": values.lingshu_i_8b,
            "fleming_vl_8b": values.fleming_vl_8b,
        },
        provenance_path=values.provenance,
        environment_root=values.environment_root,
        output=values.output,
    )
    print(json.dumps(evidence, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
