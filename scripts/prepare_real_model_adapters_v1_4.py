#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MODELS = {
    "plip": {
        "repo_id": "vinid/plip",
        "revision": "67ade53ddd32195868f422585f72698ef5d15094",
    },
    "quiltnet": {
        "repo_id": "wisdomik/QuiltNet-B-32",
        "revision": "8ce77289ce35a90b2f1db1137dfa4bc2df175e33",
    },
    "medvlm_r1": {
        "repo_id": "JZPeterPan/MedVLM-R1",
        "revision": "d256f2cfdf98c6872c1dc9f20b7dd52f49374fe9",
    },
    "biomedclip": {
        "repo_id": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "revision": "9f341de24bfb00180f1b847274256e9b65a3a32e",
    },
}
BIOMED_TEXT_MODEL = {
    "repo_id": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    "revision": "d673b8835373c6fa116d6d8006b33d48734e305d",
    "allow_patterns": [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
    ],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(directory: Path) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        relative = str(path.relative_to(directory))
        row: dict[str, Any] = {"size": path.stat().st_size}
        if path.suffix.casefold() in {".json", ".txt", ".model"} or path.name in {
            "open_clip_pytorch_model.bin",
            "model.safetensors",
            "pytorch_model.bin",
        }:
            row["sha256"] = _sha256(path)
        rows[relative] = row
    if not rows:
        raise FileNotFoundError(f"No model artifacts were downloaded to {directory}.")
    return {"files": rows, "total_bytes": sum(row["size"] for row in rows.values())}


def _download(
    *,
    repo_id: str,
    revision: str,
    directory: Path,
    allow_patterns: list[str] | None = None,
) -> dict[str, Any]:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as error:
        raise RuntimeError("Install huggingface-hub to prepare pinned model assets.") from error
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=directory,
        allow_patterns=allow_patterns,
    )
    return {
        "repo_id": repo_id,
        "revision": revision,
        "path": str(directory.resolve()),
        "source": f"https://huggingface.co/{repo_id}/tree/{revision}",
        **_manifest(directory),
    }


def prepare(asset_root: Path, selected: list[str]) -> dict[str, Any]:
    asset_root = asset_root.expanduser().resolve()
    asset_root.mkdir(parents=True, exist_ok=True)
    unknown = sorted(set(selected) - set(MODELS))
    if unknown:
        raise KeyError(f"Unknown v1.4 model assets: {', '.join(unknown)}")
    models = {
        name: _download(
            **MODELS[name],
            directory=asset_root / name.replace("_", "-"),
        )
        for name in selected
    }
    auxiliary: dict[str, Any] = {}
    if "biomedclip" in selected:
        auxiliary["biomedclip_text_model"] = _download(
            **BIOMED_TEXT_MODEL,
            directory=asset_root / "biomedclip-text-model",
        )
    result = {
        "schema_version": "1.0",
        "release": "v1.4.0",
        "models": models,
        "auxiliary_artifacts": auxiliary,
        "validation_scope": (
            "Immutable local snapshots for MedUMM adapter acceptance; downloading an "
            "artifact is not runtime validation."
        ),
    }
    (asset_root / "provenance.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare pinned MedUMM v1.4 model assets")
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=sorted(MODELS))
    return parser


def main(arguments: list[str] | None = None) -> int:
    values = build_parser().parse_args(arguments)
    print(json.dumps(prepare(values.asset_root, values.models), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
