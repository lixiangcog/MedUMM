#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


EXPECTED_REVISIONS = {
    "plip": "67ade53ddd32195868f422585f72698ef5d15094",
    "quiltnet": "8ce77289ce35a90b2f1db1137dfa4bc2df175e33",
    "medvlm_r1": "d256f2cfdf98c6872c1dc9f20b7dd52f49374fe9",
    "biomedclip": "9f341de24bfb00180f1b847274256e9b65a3a32e",
}
EXPECTED_EXECUTORS = {
    "plip": "transformers_contrastive",
    "quiltnet": "transformers_contrastive",
    "medvlm_r1": "qwen2_vl_chat",
    "biomedclip": "open_clip_hf_hub",
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
    rows = []
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if not slurm_job_id:
        raise AssertionError("Runtime acceptance must execute inside a Slurm allocation.")
    for model, path in results.items():
        result = _read_result(path)
        metadata = dict(result.get("metadata", {}))
        if result.get("model_name") != model:
            raise AssertionError(f"{model} returned the wrong model identity.")
        if metadata.get("model_revision") != EXPECTED_REVISIONS[model]:
            raise AssertionError(f"{model} did not report its pinned model revision.")
        if metadata.get("executor") != EXPECTED_EXECUTORS[model]:
            raise AssertionError(f"{model} did not use the expected explicit executor.")
        if not str(metadata.get("device", "")).startswith("cuda"):
            raise AssertionError(f"{model} did not execute on CUDA.")
        if metadata.get("scheduler", {}).get("slurm_job_id") != slurm_job_id:
            raise AssertionError(f"{model} has no matching Slurm provenance.")
        if not str(result.get("text", "")).strip():
            raise AssertionError(f"{model} produced an empty response.")
        if result.get("duration_ms") is None or float(result["duration_ms"]) <= 0:
            raise AssertionError(f"{model} has no measured inference latency.")
        scores = dict(result.get("scores", {}))
        if model != "medvlm_r1" and len(scores) != 2:
            raise AssertionError(f"{model} did not return two candidate scores.")
        asset = provenance.get("models", {}).get(model, {})
        if asset.get("revision") != EXPECTED_REVISIONS[model]:
            raise AssertionError(f"{model} asset provenance does not match the recipe.")
        environment = _environment(environment_root, model)
        if environment.get("medumm") != "1.4.0":
            raise AssertionError(f"{model} did not run the MedUMM 1.4.0 package.")
        rows.append(
            {
                "model": model,
                "model_revision": EXPECTED_REVISIONS[model],
                "executor": EXPECTED_EXECUTORS[model],
                "device": metadata["device"],
                "dtype": metadata.get("dtype"),
                "duration_ms": result["duration_ms"],
                "peak_gpu_memory_mb": metadata.get("peak_gpu_memory_mb"),
                "response": result["text"],
                "scores": scores,
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
        "release": "v1.4.0",
        "status": "passed",
        "hostname": platform.node(),
        "scheduler": {"slurm_job_id": slurm_job_id},
        "runtime": runtime,
        "validated_models": rows,
        "counts": {
            "validated_in_this_job": len(rows),
            "runtime_validated_after_release": 7,
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
    parser = argparse.ArgumentParser(description="Verify MedUMM v1.4 real model adapters")
    parser.add_argument("--plip", type=Path, required=True)
    parser.add_argument("--quiltnet", type=Path, required=True)
    parser.add_argument("--medvlm-r1", type=Path, required=True)
    parser.add_argument("--biomedclip", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(arguments: list[str] | None = None) -> int:
    values = build_parser().parse_args(arguments)
    evidence = verify(
        results={
            "plip": values.plip,
            "quiltnet": values.quiltnet,
            "medvlm_r1": values.medvlm_r1,
            "biomedclip": values.biomedclip,
        },
        provenance_path=values.provenance,
        environment_root=values.environment_root,
        output=values.output,
    )
    print(json.dumps(evidence, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
