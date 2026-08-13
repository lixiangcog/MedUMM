#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any


PATHVQA_REVISION = "1685832883334b5bb5beaf4e4b333fdeecaa4ad9"
LINGSHU_REVISION = "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"Expected a JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify(arguments: argparse.Namespace) -> dict[str, Any]:
    predictions = _jsonl(arguments.predictions)
    score = _json(arguments.score)
    audit = _json(arguments.audit)
    provenance = _json(arguments.provenance)
    if provenance.get("resolved_revision") != PATHVQA_REVISION:
        raise AssertionError("PathVQA provenance is not pinned to the accepted revision.")
    if provenance.get("answer_type_counts") != {"closed": 4, "open": 4}:
        raise AssertionError("PathVQA acceptance must contain four closed and four open questions.")
    if len(predictions) != 8 or score.get("dataset_size") != 8:
        raise AssertionError("PathVQA acceptance did not produce eight predictions.")
    if audit.get("status") not in {"passed", "warning"} or audit.get("errors"):
        raise AssertionError("PathVQA audit failed.")
    if any(not str(row.get("prediction", "")).strip() for row in predictions):
        raise AssertionError("Lingshu produced an empty PathVQA prediction.")
    metadata = [dict(row.get("model_metadata", {})) for row in predictions]
    if any(row.get("model_revision") != LINGSHU_REVISION for row in metadata):
        raise AssertionError("Lingshu revision evidence is missing or mixed.")
    if any(not str(row.get("device", "")).startswith("cuda") for row in metadata):
        raise AssertionError("PathVQA acceptance did not execute on CUDA.")
    pathology = score.get("metrics", {}).get("pathology", {})
    required = {"yes_no_accuracy", "free_form_accuracy", "overall_accuracy", "macro_answer_type_accuracy"}
    if not required <= set(pathology):
        raise AssertionError("Pathology-specific metrics are incomplete.")
    inference = score.get("metadata", {}).get("inference", {})
    if inference.get("mean_duration_ms") is None or inference.get("max_peak_gpu_memory_mb") is None:
        raise AssertionError("PathVQA runtime evidence is incomplete.")
    evidence = {
        "schema_version": "1.0",
        "release": "v1.0.0",
        "status": "passed",
        "scheduler": {"verification_job_id": os.environ.get("SLURM_JOB_ID")},
        "hostname": platform.node(),
        "runtime": {
            "python": platform.python_version(),
        },
        "slice": {
            "model": "lingshu_7b",
            "model_revision": LINGSHU_REVISION,
            "dataset": "pathvqa",
            "dataset_revision": PATHVQA_REVISION,
            "sample_count": 8,
            "answer_type_counts": provenance["answer_type_counts"],
            "metrics": score["metrics"],
            "inference": inference,
            "audit_status": audit["status"],
        },
        "metric_suites": [
            "pathology_vqa",
            "medical_report_factuality",
            "medical_grounding",
            "medical_measurement",
            "medical_calibration",
        ],
        "validation_scope": (
            "Pinned PathVQA pathology VQA wiring plus deterministic specialized-metric tests; "
            "not a clinical performance or full-dataset quality claim."
        ),
    }
    try:
        import torch

        evidence["runtime"].update({
            "torch": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    except ImportError:
        evidence["runtime"]["torch"] = None
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return evidence


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify MedUMM v1.0 clinical evaluation")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    values = parser.parse_args(arguments)
    print(json.dumps(verify(values), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
