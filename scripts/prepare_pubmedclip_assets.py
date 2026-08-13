#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MODEL_ID = "flaviagiammarino/pubmed-clip-vit-base-patch32"
MODEL_REVISION = "26c0c67f6da303ad2a38909130bd35744ea93517"
REQUIRED_FILES = (
    "config.json",
    "preprocessor_config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def prepare_pubmedclip_assets(destination: Path) -> dict[str, Any]:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as error:
        raise RuntimeError("Install huggingface-hub to prepare PubMedCLIP.") from error
    model_directory = destination.expanduser().resolve() / "pubmedclip"
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=model_directory,
        allow_patterns=list(REQUIRED_FILES),
    )
    missing = [name for name in REQUIRED_FILES if not (model_directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete PubMedCLIP snapshot: {missing}")
    files = {
        name: {
            "size": (model_directory / name).stat().st_size,
            "sha256": _digest(model_directory / name),
        }
        for name in REQUIRED_FILES
    }
    provenance = {
        "schema_version": "1.0",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_path": str(model_directory),
        "files": files,
        "source": f"https://huggingface.co/{MODEL_ID}",
        "clinical_use": False,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare pinned PubMedCLIP assets")
    parser.add_argument("--destination", type=Path, required=True)
    return parser


def main(arguments: list[str] | None = None) -> int:
    values = build_parser().parse_args(arguments)
    print(
        json.dumps(
            prepare_pubmedclip_assets(values.destination),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
