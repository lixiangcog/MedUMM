#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(arguments: argparse.Namespace) -> dict:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("prepare_alignment_assets requires huggingface_hub.") from error
    output = arguments.output_directory
    output.mkdir(parents=True, exist_ok=True)
    local = output / "model"
    if not arguments.local_only:
        snapshot_download(
            repo_id=arguments.model_id,
            revision=arguments.revision,
            local_dir=local,
            token=os.environ.get("HF_TOKEN"),
            allow_patterns=[
                "*.json",
                "*.model",
                "*.safetensors",
                "tokenizer*",
                "merges.txt",
                "vocab.json",
            ],
        )
    if not (local / "config.json").is_file() or not any(
        local.glob("*.safetensors")
    ):
        raise FileNotFoundError("Alignment asset snapshot lacks config or safetensors weights.")
    files = [
        path
        for path in sorted(local.rglob("*"))
        if path.is_file()
        and ".cache" not in path.parts
        and not any(
            part.startswith(".") and ".parts-" in part for part in path.parts
        )
    ]
    provenance = {
        "schema_version": "1.0",
        "model_id": arguments.model_id,
        "model_revision": arguments.revision,
        "license": arguments.license,
        "local_path": str(local),
        "file_count": len(files),
        "files": [
            {
                "path": str(path.relative_to(local)),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
        "clinical_use": False,
    }
    path = output / "provenance.json"
    path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))
    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare pinned v0.7 alignment model assets")
    parser.add_argument("--model-id", default="EleutherAI/pythia-14m")
    parser.add_argument(
        "--revision", default="cf967c0a9a04383db6f7b1108d86b2962634b4ac"
    )
    parser.add_argument("--license", default="Apache-2.0")
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--local-only", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(0 if prepare(build_parser().parse_args()) else 1)
