#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DATASET = "flaviagiammarino/vqa-rad"
DEFAULT_REVISION = "bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9"
REAL_TASKS = (
    "finding_assessment",
    "clinical_description",
    "anatomy_localization",
    "quantitative_assessment",
    "image_context",
    "diagnostic_reasoning",
)


def classify_question(question: str, answer: str) -> tuple[str, str]:
    """Map VQA-RAD questions transparently; this is not an expert task label."""

    text = " ".join(question.casefold().split())
    if re.search(
        r"\bdiagnos|\bdifferential|\b(?:what|which) disease\b|"
        r"\bmost likely (?:diagnosis|condition)\b|\bcondition causing\b|"
        r"\bthe condition in which\b|\bwhat is the condition\b|"
        r"\bconsistent with what condition\b|\bwhat condition does\b",
        text,
    ):
        return "diagnostic_reasoning", "diagnosis_pattern"
    if re.search(
        r"\bmodality\b|\btype of (?:image|scan)\b|\bkind of (?:image|scan)\b|"
        r"\bimaging technique\b|\bhow (?:was|is) (?:this|the) image taken\b|"
        r"\b(?:axial|coronal|sagittal) (?:image|view|plane)\b|\bcontrast (?:used|enhanced)\b|"
        r"\bwhat (?:view|plane)\b",
        text,
    ):
        return "image_context", "image_context_pattern"
    if re.search(
        r"\bhow (?:large|big|many|much)\b|\b(?:size|measure|measurement|diameter|number of)\b|"
        r"\b(?:larger|smaller)\b",
        text,
    ):
        return "quantitative_assessment", "quantity_pattern"
    if re.search(
        r"\bwhere\b|\blocation\b|\blocated\b|\bwhich side\b|\bwhat side\b|"
        r"\bwhat (?:part|region|organ)\b",
        text,
    ):
        return "anatomy_localization", "location_pattern"
    if answer.casefold().strip() in {"yes", "no"} or re.match(
        r"^(?:is|are|does|do|has|have|can|could|was|were)\b", text
    ):
        return "finding_assessment", "closed_finding_pattern"
    return "clinical_description", "open_description_fallback"


def _record_id(split: str, index: int, question: str) -> str:
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:8]
    return f"vqa-rad-task-{split}-{index:05d}-{digest}"


def _case_id(image: Any) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(str(rgb.size).encode())
    digest.update(rgb.tobytes())
    return f"vqa-rad-case-{digest.hexdigest()[:16]}"


def export_vqa_rad_tasks(
    *,
    dataset_name: str,
    revision: str,
    split: str,
    output_directory: Path,
    samples_per_task: int,
    parquet_path: Path | None = None,
) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as error:
        raise RuntimeError("Install MedUMM with the 'data' extra to prepare VQA-RAD.") from error

    if samples_per_task < 1:
        raise ValueError("samples_per_task must be at least one.")
    if parquet_path is not None:
        dataset = load_dataset(
            "parquet",
            data_files={split: str(parquet_path.expanduser().resolve())},
            split=split,
        )
    else:
        dataset = load_dataset(dataset_name, split=split, revision=revision)

    selected: dict[str, list[tuple[int, Any, str, str, int]]] = defaultdict(list)
    source_distribution: Counter[str] = Counter()
    rule_distribution: Counter[str] = Counter()
    case_turns: Counter[str] = Counter()
    for source_index, sample in enumerate(dataset):
        question = str(sample.get("question", "")).strip()
        answer = str(sample.get("answer", "")).strip()
        if not question or not answer or sample.get("image") is None:
            continue
        task, rule = classify_question(question, answer)
        case_id = _case_id(sample["image"])
        case_turns[case_id] += 1
        source_distribution[task] += 1
        rule_distribution[rule] += 1
        if len(selected[task]) < samples_per_task:
            selected[task].append(
                (source_index, sample, rule, case_id, case_turns[case_id])
            )

    missing = [task for task in REAL_TASKS if len(selected[task]) < samples_per_task]
    if missing:
        raise ValueError(f"VQA-RAD cannot supply the requested balanced tasks: {missing}")

    images_directory = output_directory / "images"
    images_directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for task in REAL_TASKS:
        for source_index, sample, rule, case_id, turn_index in selected[task]:
            question = str(sample["question"]).strip()
            answer = str(sample["answer"]).strip()
            sample_id = _record_id(split, source_index, question)
            image_name = f"{case_id}.png"
            sample["image"].convert("RGB").save(images_directory / image_name)
            is_closed = answer.casefold() in {"yes", "no"}
            row: dict[str, Any] = {
                "id": sample_id,
                "task": task,
                "prompt": question,
                "image": image_name,
                "references": [answer],
                "answer_type": "closed" if is_closed else "open",
                "specialty": "radiology",
                "modality": "radiology",
                "anatomy": "unknown",
                "concepts": [] if is_closed else [answer],
                "evidence": [],
                "language": "en",
                "case_id": case_id,
                "turn_index": turn_index,
                "reference_provenance": {
                    "kind": "dataset_answer",
                    "dataset": dataset_name,
                    "source_index": source_index,
                },
                "metadata": {
                    "source_index": source_index,
                    "source_split": split,
                    "task_mapping": {
                        "method": "heuristic",
                        "version": "vqa_rad_question_rules_v1",
                        "rule": rule,
                        "expert_validated": False,
                    },
                },
            }
            if is_closed:
                row["choices"] = {"A": "yes", "B": "no"}
            rows.append(row)

    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "samples.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    provenance = {
        "schema_version": "1.0",
        "dataset": dataset_name,
        "requested_revision": revision,
        "resolved_revision": revision,
        "split": split,
        "sample_count": len(rows),
        "selection": {
            "method": "balanced_first_in_source_order",
            "samples_per_task": samples_per_task,
            "tasks": list(REAL_TASKS),
            "unique_cases": len({row["case_id"] for row in rows}),
            "multi_turn_cases": len(
                {
                    row["case_id"]
                    for row in rows
                    if sum(other["case_id"] == row["case_id"] for other in rows) > 1
                }
            ),
        },
        "task_mapping": {
            "method": "heuristic",
            "version": "vqa_rad_question_rules_v1",
            "expert_validated": False,
            "source_distribution": dict(sorted(source_distribution.items())),
            "selected_distribution": dict(
                sorted(Counter(row["task"] for row in rows).items())
            ),
            "rule_distribution": dict(sorted(rule_distribution.items())),
        },
        "license": "CC0-1.0",
        "source": "https://huggingface.co/datasets/flaviagiammarino/vqa-rad",
        "source_parquet": str(parquet_path.resolve()) if parquet_path else None,
        "clinical_use": False,
        "manifest": str(manifest_path),
    }
    (output_directory / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a balanced, task-aware VQA-RAD slice for MedUMM v0.6"
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--parquet-path", type=Path)
    parser.add_argument("--samples-per-task", type=int, default=4)
    return parser


def main(arguments: list[str] | None = None) -> int:
    values = build_parser().parse_args(arguments)
    provenance = export_vqa_rad_tasks(
        dataset_name=values.dataset,
        revision=values.revision,
        split=values.split,
        output_directory=values.output_directory,
        samples_per_task=values.samples_per_task,
        parquet_path=values.parquet_path,
    )
    print(json.dumps(provenance, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
