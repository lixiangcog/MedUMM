#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

from scripts.prepare_medical_vqa_dataset import export_medical_vqa_dataset
from scripts.prepare_medmnist_classification import export_medmnist_classification
from scripts.prepare_pubmedclip_assets import prepare_pubmedclip_assets


LINGSHU_ID = "lingshu-medical-mllm/Lingshu-7B"
LINGSHU_REVISION = "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9"
SLAKE_ID = "BoKelvin/SLAKE"
SLAKE_REVISION = "a9083ce6c34ac3ffb17671a605962924d8a8f9e9"
PNEUMONIAMNIST_URL = (
    "https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1"
)
LINGSHU_FILES = (
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _download_lingshu(asset_root: Path) -> dict[str, Any]:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as error:
        raise RuntimeError("Install huggingface-hub to prepare Lingshu.") from error
    model_directory = asset_root / "lingshu-7b"
    snapshot_download(
        repo_id=LINGSHU_ID,
        revision=LINGSHU_REVISION,
        local_dir=model_directory,
        allow_patterns=list(LINGSHU_FILES),
    )
    missing = [name for name in LINGSHU_FILES if not (model_directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete Lingshu snapshot: {missing}")
    provenance = {
        "schema_version": "1.0",
        "model_id": LINGSHU_ID,
        "model_revision": LINGSHU_REVISION,
        "model_path": str(model_directory.resolve()),
        "files": {
            name: {"size": (model_directory / name).stat().st_size}
            for name in LINGSHU_FILES
        },
        "source": f"https://huggingface.co/{LINGSHU_ID}",
        "clinical_use": False,
    }
    (asset_root / "lingshu-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return provenance


def _download_slake(asset_root: Path, data_root: Path, sample_count: int) -> dict[str, Any]:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as error:
        raise RuntimeError("Install huggingface-hub to prepare SLAKE.") from error
    snapshot = Path(
        snapshot_download(
            repo_id=SLAKE_ID,
            repo_type="dataset",
            revision=SLAKE_REVISION,
            local_dir=asset_root / "slake-source",
            allow_patterns=["test.json", "imgs.zip"],
        )
    )
    source_json = snapshot / "test.json"
    images_root = asset_root / "slake-images"
    marker = images_root / ".complete"
    if not marker.is_file():
        if images_root.exists():
            shutil.rmtree(images_root)
        images_root.mkdir(parents=True)
        with zipfile.ZipFile(snapshot / "imgs.zip") as archive:
            archive.extractall(images_root)
        marker.write_text(SLAKE_REVISION + "\n", encoding="utf-8")
    candidates = [images_root / "imgs", images_root]
    image_root = next((path for path in candidates if (path / "xmlab1").is_dir()), None)
    if image_root is None:
        raise FileNotFoundError("SLAKE imgs.zip did not contain the expected xmlab directories.")
    return export_medical_vqa_dataset(
        dataset="slake",
        revision=SLAKE_REVISION,
        split="test",
        output_directory=data_root / "slake",
        max_samples=sample_count,
        language="en",
        source_path=source_json,
        image_root=image_root,
    )


def _download_pneumoniamnist(
    asset_root: Path, data_root: Path, sample_count: int
) -> dict[str, Any]:
    archive = asset_root / "pneumoniamnist.npz"
    if not archive.is_file() or archive.stat().st_size == 0:
        partial = archive.with_suffix(".npz.partial")
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "10",
                "--connect-timeout",
                "30",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                PNEUMONIAMNIST_URL,
            ],
            check=True,
        )
        partial.replace(archive)
    return export_medmnist_classification(
        dataset="pneumoniamnist",
        revision="v2",
        split="test",
        npz_path=archive,
        output_directory=data_root / "pneumoniamnist",
        max_samples=sample_count,
    )


def prepare(
    *, asset_root: Path, data_root: Path, slake_samples: int, classification_samples: int
) -> dict[str, Any]:
    asset_root = asset_root.expanduser().resolve()
    data_root = data_root.expanduser().resolve()
    asset_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    evidence = {
        "schema_version": "1.0",
        "release": "v0.9.0",
        "models": {
            "lingshu_7b": _download_lingshu(asset_root),
            "pubmedclip": prepare_pubmedclip_assets(asset_root),
        },
        "datasets": {
            "slake": _download_slake(asset_root, data_root, slake_samples),
            "pneumoniamnist": _download_pneumoniamnist(
                asset_root, data_root, classification_samples
            ),
        },
    }
    (asset_root / "provenance.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare pinned MedUMM v0.9 runtime slices")
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--slake-samples", type=int, default=4)
    parser.add_argument("--classification-samples", type=int, default=32)
    return parser


def main(arguments: list[str] | None = None) -> int:
    values = build_parser().parse_args(arguments)
    print(
        json.dumps(
            prepare(
                asset_root=values.asset_root,
                data_root=values.data_root,
                slake_samples=values.slake_samples,
                classification_samples=values.classification_samples,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
