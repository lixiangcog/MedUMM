#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_DATASET = "flaviagiammarino/vqa-rad"


def _record_id(split: str, index: int, question: str) -> str:
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:8]
    return f"vqa-rad-{split}-{index:05d}-{digest}"


def _dataset_revision(dataset_name: str, revision: str) -> str:
    try:
        from huggingface_hub import HfApi

        return str(HfApi().dataset_info(dataset_name, revision=revision).sha)
    except Exception:
        return revision


def export_vqa_rad(
    *,
    dataset_name: str,
    revision: str,
    split: str,
    output_directory: Path,
    max_samples: int,
    closed_only: bool,
    parquet_path: Path | None = None,
    csv_path: Path | None = None,
    json_path: Path | None = None,
    image_root: Path | None = None,
    source_mirror: str | None = None,
    source_artifact_sha256: str | None = None,
    canonical_revision: str | None = None,
) -> dict[str, Any]:
    source_url = source_mirror or (
        "https://osf.io/89kps/" if json_path is not None else f"https://huggingface.co/datasets/{dataset_name}"
    )
    local_inputs = sum(path is not None for path in (parquet_path, csv_path, json_path))
    if local_inputs > 1:
        raise ValueError("Use only one of parquet_path, csv_path, or json_path.")
    if csv_path is not None or json_path is not None:
        if image_root is None:
            raise ValueError("--image-root is required with --csv-path or --json-path.")
        resolved_image_root = image_root.expanduser().resolve()
        if csv_path is not None:
            resolved_csv = csv_path.expanduser().resolve()
            with resolved_csv.open(newline="", encoding="utf-8-sig") as stream:
                dataset: Any = list(csv.DictReader(stream))
        else:
            resolved_json = json_path.expanduser().resolve()
            loaded = json.loads(resolved_json.read_text(encoding="utf-8-sig"))
            if not isinstance(loaded, list) or not all(
                isinstance(item, dict) for item in loaded
            ):
                raise ValueError("VQA-RAD source JSON must be a list of objects.")
            dataset = loaded
    else:
        try:
            from datasets import load_dataset
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Install MedUMM with the 'data' extra to prepare VQA-RAD."
            ) from error
        if parquet_path is not None:
            dataset = load_dataset(
                "parquet",
                data_files={split: str(parquet_path.expanduser().resolve())},
                split=split,
            )
        else:
            dataset = load_dataset(dataset_name, split=split, revision=revision)
    images_directory = output_directory / "images"
    images_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    smoke_image_written = False
    for source_index, sample in enumerate(dataset):
        question = str(sample.get("question", "")).strip()
        answer = str(sample.get("answer", "")).strip()
        source_image = sample.get("image")
        if csv_path is not None or json_path is not None:
            source_image = resolved_image_root / str(
                sample.get("img_id", sample.get("image_name", ""))
            )
        if not question or not answer or source_image is None:
            continue
        normalized = answer.casefold()
        is_closed = normalized in {"yes", "no"}
        if closed_only and not is_closed:
            continue
        sample_id = _record_id(split, source_index, question)
        image_name = f"{sample_id}.png"
        if isinstance(source_image, Path):
            if not source_image.is_file():
                raise FileNotFoundError(f"VQA-RAD image is missing: {source_image}")
            from PIL import Image

            with Image.open(source_image) as opened:
                image = opened.convert("RGB")
        else:
            image = source_image.convert("RGB")
        image.save(images_directory / image_name)
        if not smoke_image_written:
            image.save(images_directory / "smoke.png")
            smoke_image_written = True
        row: dict[str, Any] = {
            "id": sample_id,
            "image": image_name,
            "question": question,
            "answer": answer,
            "answer_type": "closed" if is_closed else "open",
            "modality": "radiology",
            "category": str(
                sample.get("question_type", sample.get("category", "radiology"))
            ),
            "language": "en",
            "metadata": {
                "source_index": source_index,
                "source_split": split,
                "source_qid": sample.get("qid"),
                "source_image_sha256": hashlib.sha256(source_image.read_bytes()).hexdigest()
                if isinstance(source_image, Path)
                else None,
            },
        }
        if is_closed:
            row["choices"] = {"A": "yes", "B": "no"}
        rows.append(row)
        if max_samples and len(rows) >= max_samples:
            break
    if not rows:
        raise ValueError("No VQA-RAD samples matched the export selection.")

    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "samples.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    resolved_revision = (
        revision
        if parquet_path is not None or csv_path is not None or json_path is not None
        else _dataset_revision(dataset_name, revision)
    )
    provenance = {
        "schema_version": "1.0",
        "dataset": dataset_name,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "split": split,
        "sample_count": len(rows),
        "selection": {"closed_only": closed_only, "max_samples": max_samples},
        "license": "CC0-1.0",
        "source": source_url,
        "source_parquet": str(parquet_path.resolve()) if parquet_path else None,
        "source_csv": str(csv_path.resolve()) if csv_path else None,
        "source_json": str(json_path.resolve()) if json_path else None,
        "source_image_root": str(image_root.resolve()) if image_root else None,
        "source_mirror": source_mirror,
        "source_artifact_sha256": source_artifact_sha256,
        "canonical_revision": canonical_revision,
        "clinical_use": False,
        "manifest": str(manifest_path),
        "smoke_image": str(images_directory / "smoke.png"),
    }
    (output_directory / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export VQA-RAD to the MedUMM JSONL schema")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--parquet-path", type=Path)
    parser.add_argument("--csv-path", type=Path)
    parser.add_argument("--json-path", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--source-mirror")
    parser.add_argument("--source-artifact-sha256")
    parser.add_argument("--canonical-revision")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--closed-only", action="store_true")
    return parser


def main(arguments: list[str] | None = None) -> int:
    values = build_parser().parse_args(arguments)
    provenance = export_vqa_rad(
        dataset_name=values.dataset,
        revision=values.revision,
        split=values.split,
        output_directory=values.output_directory,
        max_samples=values.max_samples,
        closed_only=values.closed_only,
        parquet_path=values.parquet_path,
        csv_path=values.csv_path,
        json_path=values.json_path,
        image_root=values.image_root,
        source_mirror=values.source_mirror,
        source_artifact_sha256=values.source_artifact_sha256,
        canonical_revision=values.canonical_revision,
    )
    print(json.dumps(provenance, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
