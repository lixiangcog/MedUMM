#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


DATASETS = {
    "slake": {
        "artifact_id": "BoKelvin/SLAKE",
        "license": "CC-BY-4.0",
        "source": "https://huggingface.co/datasets/BoKelvin/SLAKE",
        "default_split": "test",
        "modality": "radiology",
    },
    "pathvqa": {
        "artifact_id": "flaviagiammarino/path-vqa",
        "license": "MIT (dataset card); underlying images retain source copyrights",
        "source": "https://huggingface.co/datasets/flaviagiammarino/path-vqa",
        "default_split": "test",
        "modality": "pathology",
    },
}


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _stable_id(dataset: str, split: str, index: int, question: str) -> str:
    suffix = hashlib.sha256(question.encode("utf-8")).hexdigest()[:8]
    return f"{dataset}-{split}-{index:05d}-{suffix}"


def _load_local_records(path: Path, split: str) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ModuleNotFoundError as error:
            raise RuntimeError("Install MedUMM with the 'data' extra for parquet input.") from error
        return parquet.read_table(path).to_pylist()
    loaded = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(loaded, dict):
        loaded = loaded.get(split, loaded.get("data"))
    if not isinstance(loaded, list) or not all(isinstance(item, dict) for item in loaded):
        raise ValueError(f"Expected a list of records in {path}.")
    return loaded


def _load_records(
    *,
    dataset_id: str,
    revision: str,
    split: str,
    source_path: Path | None,
    streaming: bool = False,
) -> tuple[Iterable[dict[str, Any]], str]:
    if source_path is not None:
        return _load_local_records(source_path, split), revision
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ModuleNotFoundError as error:
        raise RuntimeError("Install MedUMM with the 'data' extra for remote datasets.") from error
    resolved = str(HfApi().dataset_info(dataset_id, revision=revision).sha)
    return load_dataset(
        dataset_id,
        split=split,
        revision=resolved,
        streaming=streaming,
    ), resolved


def _language(record: dict[str, Any], dataset: str) -> str:
    value = record.get("q_lang", record.get("language", "en"))
    normalized = str(value).strip().casefold()
    if dataset == "slake" and normalized in {"zh", "cn", "chinese"}:
        return "zh"
    return "en" if normalized in {"", "english", "en"} else normalized


def _source_image(record: dict[str, Any], dataset: str, image_root: Path | None) -> Any:
    value = record.get("image")
    if value is not None and not isinstance(value, str):
        return value
    if image_root is None:
        return None
    if dataset == "slake":
        relative = record.get("img_name", value)
    else:
        relative = record.get("image_name", record.get("img_name", value))
    if relative in {None, ""}:
        return None
    return image_root / str(relative)


def export_medical_vqa_dataset(
    *,
    dataset: str,
    revision: str,
    split: str,
    output_directory: Path,
    max_samples: int = 0,
    language: str = "en",
    closed_only: bool = False,
    closed_samples: int = 0,
    open_samples: int = 0,
    streaming: bool = False,
    source_path: Path | None = None,
    image_root: Path | None = None,
) -> dict[str, Any]:
    name = dataset.strip().casefold()
    if name not in DATASETS:
        raise ValueError(f"Unsupported medical VQA dataset: {dataset!r}.")
    spec = DATASETS[name]
    if revision.casefold() in {"", "main", "master", "latest", "head"}:
        raise ValueError("A pinned immutable dataset revision is required.")
    if min(closed_samples, open_samples) < 0:
        raise ValueError("Answer-type sample quotas cannot be negative.")
    if closed_only and open_samples:
        raise ValueError("closed_only cannot be combined with an open sample quota.")
    resolved_source = source_path.expanduser().resolve() if source_path else None
    resolved_images = image_root.expanduser().resolve() if image_root else None
    records, resolved_revision = _load_records(
        dataset_id=spec["artifact_id"],
        revision=revision,
        split=split,
        source_path=resolved_source,
        streaming=streaming,
    )
    images_directory = output_directory / "images"
    images_directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    selected = {"closed": 0, "open": 0}
    skipped = {"language": 0, "incomplete": 0, "missing_image": 0, "answer_type": 0}
    for source_index, record in enumerate(records):
        record_language = _language(record, name)
        if language != "all" and record_language != language:
            skipped["language"] += 1
            continue
        question = str(record.get("question", "")).strip()
        answer = str(record.get("answer", "")).strip()
        if not question or not answer:
            skipped["incomplete"] += 1
            continue
        normalized_answer = answer.casefold()
        declared_type = str(record.get("answer_type", "")).strip().casefold()
        is_closed = normalized_answer in {"yes", "no"} or declared_type in {
            "closed", "binary", "yes/no", "yes_no",
        }
        if closed_only and not is_closed:
            skipped["answer_type"] += 1
            continue
        answer_type = "closed" if is_closed else "open"
        quota = closed_samples if is_closed else open_samples
        if quota and selected[answer_type] >= quota:
            skipped["answer_type"] += 1
            continue
        source_image = _source_image(record, name, resolved_images)
        if source_image is None:
            skipped["missing_image"] += 1
            continue
        from PIL import Image

        source_hash = None
        if isinstance(source_image, Path):
            if not source_image.is_file():
                skipped["missing_image"] += 1
                continue
            source_hash = _digest(source_image)
            opened = Image.open(source_image)
        else:
            opened = source_image
        sample_id = _stable_id(name, split, source_index, question)
        image_name = f"{sample_id}.png"
        try:
            opened.convert("RGB").save(images_directory / image_name)
        finally:
            close = getattr(opened, "close", None)
            if callable(close):
                close()
        if len(rows) == 0:
            with Image.open(images_directory / image_name) as smoke:
                smoke.save(images_directory / "smoke.png")
        category = str(
            record.get("content_type", record.get("question_type", record.get("category", name)))
        ).strip()
        row: dict[str, Any] = {
            "id": sample_id,
            "image": image_name,
            "question": question,
            "answer": answer,
            "answer_type": answer_type,
            "modality": str(record.get("modality", spec["modality"])).strip().casefold(),
            "category": category or name,
            "language": record_language,
            "metadata": {
                "source_index": source_index,
                "source_split": split,
                "source_qid": record.get("qid", record.get("question_id")),
                "source_image_id": record.get("img_id", record.get("image_id")),
                "source_image_sha256": source_hash,
                "source_location": record.get("location"),
                "source_base_type": record.get("base_type"),
            },
        }
        if is_closed:
            row["choices"] = {"A": "yes", "B": "no"}
        rows.append(row)
        selected[answer_type] += 1
        quotas_complete = (
            (not closed_samples or selected["closed"] >= closed_samples)
            and (not open_samples or selected["open"] >= open_samples)
        )
        if (closed_samples or open_samples) and quotas_complete:
            break
        if not (closed_samples or open_samples) and max_samples and len(rows) >= max_samples:
            break
    if not rows:
        raise ValueError(f"No {name} samples matched the export selection.")
    if closed_samples and selected["closed"] < closed_samples:
        raise ValueError(
            f"Requested {closed_samples} closed samples but only found {selected['closed']}."
        )
    if open_samples and selected["open"] < open_samples:
        raise ValueError(
            f"Requested {open_samples} open samples but only found {selected['open']}."
        )
    manifest_path = output_directory / "samples.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    provenance = {
        "schema_version": "1.0",
        "dataset": name,
        "artifact_id": spec["artifact_id"],
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "split": split,
        "sample_count": len(rows),
        "selection": {
            "max_samples": max_samples,
            "language": language,
            "closed_only": closed_only,
            "closed_samples": closed_samples,
            "open_samples": open_samples,
            "streaming": streaming,
        },
        "answer_type_counts": selected,
        "skipped": skipped,
        "license": spec["license"],
        "source": spec["source"],
        "source_path": str(resolved_source) if resolved_source else None,
        "source_path_sha256": _digest(resolved_source) if resolved_source else None,
        "source_image_root": str(resolved_images) if resolved_images else None,
        "manifest": str(manifest_path.resolve()),
        "clinical_use": False,
    }
    (output_directory / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export SLAKE or PathVQA to MedUMM JSONL")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--revision", required=True)
    parser.add_argument("--split")
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--language", choices=("en", "zh", "all"), default="en")
    parser.add_argument("--closed-only", action="store_true")
    parser.add_argument("--closed-samples", type=int, default=0)
    parser.add_argument("--open-samples", type=int, default=0)
    parser.add_argument("--streaming", action="store_true")
    return parser


def main(arguments: list[str] | None = None) -> int:
    values = build_parser().parse_args(arguments)
    spec = DATASETS[values.dataset]
    result = export_medical_vqa_dataset(
        dataset=values.dataset,
        revision=values.revision,
        split=values.split or spec["default_split"],
        output_directory=values.output_directory,
        max_samples=values.max_samples,
        language=values.language,
        closed_only=values.closed_only,
        closed_samples=values.closed_samples,
        open_samples=values.open_samples,
        streaming=values.streaming,
        source_path=values.source_path,
        image_root=values.image_root,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
