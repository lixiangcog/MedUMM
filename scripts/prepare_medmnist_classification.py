#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DATASETS = {
    "pneumoniamnist": {
        "labels": {0: "normal chest x-ray", 1: "pneumonia chest x-ray"},
        "modality": "chest_xray",
        "category": "pneumonia_classification",
        "license": "CC-BY-4.0",
        "source": (
            "https://zenodo.org/records/10519652/files/"
            "pneumoniamnist.npz?download=1"
        ),
        "md5": "28209eda62fecd6e6a2d98b1501bb15f",
    },
}


def _digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def export_medmnist_classification(
    *,
    dataset: str,
    revision: str,
    split: str,
    npz_path: Path,
    output_directory: Path,
    max_samples: int = 0,
) -> dict[str, Any]:
    name = dataset.strip().casefold()
    if name not in DATASETS:
        raise ValueError(f"Unsupported MedMNIST dataset: {dataset!r}.")
    if revision != "v2":
        raise ValueError("The current MedMNIST exporter requires the fixed v2 release.")
    if split not in {"train", "val", "test"}:
        raise ValueError("MedMNIST split must be train, val, or test.")
    source = npz_path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"MedMNIST archive not found: {source}")
    spec = DATASETS[name]
    source_md5 = _digest(source, "md5")
    if source_md5 != spec["md5"]:
        raise ValueError(
            f"MedMNIST archive MD5 mismatch: expected {spec['md5']}, found {source_md5}."
        )
    try:
        import numpy as np
        from PIL import Image
    except ModuleNotFoundError as error:
        raise RuntimeError("Install MedUMM with the 'baseline' extra.") from error
    with np.load(source) as archive:
        images = archive[f"{split}_images"]
        labels = archive[f"{split}_labels"].reshape(-1)
        if len(images) != len(labels):
            raise ValueError("MedMNIST image and label counts differ.")
        limit = min(len(images), max_samples) if max_samples else len(images)
        images_directory = output_directory / "images"
        images_directory.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        candidates = {chr(65 + key): value for key, value in spec["labels"].items()}
        for index in range(limit):
            label_id = int(labels[index])
            if label_id not in spec["labels"]:
                raise ValueError(f"Unexpected {name} label: {label_id}")
            sample_id = f"{name}-{split}-{index:05d}"
            image_name = f"{sample_id}.png"
            Image.fromarray(images[index]).convert("RGB").save(images_directory / image_name)
            if index == 0:
                with Image.open(images_directory / image_name) as smoke:
                    smoke.save(images_directory / "smoke.png")
            answer = str(spec["labels"][label_id])
            rows.append(
                {
                    "id": sample_id,
                    "image": image_name,
                    "question": "Classify this pediatric chest X-ray.",
                    "answer": answer,
                    "choices": candidates,
                    "answer_type": "multiple_choice",
                    "modality": spec["modality"],
                    "category": spec["category"],
                    "language": "en",
                    "metadata": {
                        "source_index": index,
                        "source_split": split,
                        "source_label_id": label_id,
                        "original_resolution": list(images[index].shape),
                    },
                }
            )
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "samples.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    provenance = {
        "schema_version": "1.0",
        "dataset": name,
        "artifact_id": "MedMNIST/PneumoniaMNIST-v2",
        "requested_revision": revision,
        "resolved_revision": revision,
        "split": split,
        "sample_count": len(rows),
        "selection": {"max_samples": max_samples},
        "labels": spec["labels"],
        "license": spec["license"],
        "source": spec["source"],
        "source_archive": str(source),
        "source_archive_md5": source_md5,
        "source_archive_sha256": _digest(source),
        "manifest": str(manifest_path.resolve()),
        "clinical_use": False,
    }
    (output_directory / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export MedMNIST classification data to the MedUMM JSONL schema"
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--npz-path", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser


def main(arguments: list[str] | None = None) -> int:
    values = build_parser().parse_args(arguments)
    result = export_medmnist_classification(
        dataset=values.dataset,
        revision=values.revision,
        split=values.split,
        npz_path=values.npz_path,
        output_directory=values.output_directory,
        max_samples=values.max_samples,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
