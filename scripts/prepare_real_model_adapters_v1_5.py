#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any


MODELS = {
    "medmo_4b": {
        "repo_id": "MBZUAI/MedMO-4B",
        "revision": "0e220705d851598b37725326aadb852aa8b37f43",
    },
    "medmo_8b": {
        "repo_id": "MBZUAI/MedMO-8B",
        "revision": "8eafab80545fb4d60b0fb126a097e972e6475851",
    },
    "lingshu_i_8b": {
        "repo_id": "lingshu-medical-mllm/Lingshu-I-8B",
        "revision": "b004bfc0554d90bd44baedf4de08c361e71ef017",
    },
    "fleming_vl_8b": {
        "repo_id": "UbiquantAI/Fleming-VL-8B",
        "revision": "801e2bef9645bca0646d55837a6630fb468e2901",
    },
}

ACCESS_BLOCKED = {
    "medsiglip": {
        "repo_id": "google/medsiglip-448",
        "revision": "9cea28a1a1195f665105faa6e8544c112fd960a4",
        "access": "gated",
    },
    "medgemma_1_5_4b_it": {
        "repo_id": "google/medgemma-1.5-4b-it",
        "revision": "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b",
        "access": "gated",
    },
    "maira_2": {
        "repo_id": "microsoft/maira-2",
        "revision": "795a2b1cd4a310624b4e3d14b5a23e41fd273deb",
        "access": "gated",
    },
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
        if path.suffix.casefold() in {
            ".bin",
            ".json",
            ".model",
            ".py",
            ".safetensors",
            ".txt",
        }:
            row["sha256"] = _sha256(path)
        rows[relative] = row
    if not rows:
        raise FileNotFoundError(f"No model artifacts were downloaded to {directory}.")
    return {"files": rows, "total_bytes": sum(row["size"] for row in rows.values())}


def _download(*, repo_id: str, revision: str, directory: Path) -> dict[str, Any]:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as error:
        raise RuntimeError("Install huggingface-hub to prepare pinned model assets.") from error
    for attempt in range(1, 7):
        try:
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=directory,
                max_workers=8,
            )
            break
        except Exception:
            if attempt == 6:
                raise
            time.sleep(min(5 * attempt, 30))
    return {
        "repo_id": repo_id,
        "revision": revision,
        "path": str(directory.resolve()),
        "source": f"https://huggingface.co/{repo_id}/tree/{revision}",
        **_manifest(directory),
    }


def _probe_gated_access() -> dict[str, Any]:
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError
    except ModuleNotFoundError as error:
        raise RuntimeError("Install huggingface-hub to probe gated model access.") from error
    results: dict[str, Any] = {}
    for name, spec in ACCESS_BLOCKED.items():
        try:
            hf_hub_download(
                repo_id=spec["repo_id"],
                filename="config.json",
                revision=spec["revision"],
            )
            results[name] = {**spec, "status": "authorized"}
        except GatedRepoError:
            results[name] = {
                **spec,
                "status": "blocked",
                "reason": "upstream_terms_or_authorized_token_required",
            }
        except Exception as error:  # network failures must not be mislabeled as access denial
            results[name] = {
                **spec,
                "status": "probe_unavailable",
                "reason": type(error).__name__,
            }
    return results


def prepare(asset_root: Path, selected: list[str]) -> dict[str, Any]:
    asset_root = asset_root.expanduser().resolve()
    asset_root.mkdir(parents=True, exist_ok=True)
    unknown = sorted(set(selected) - set(MODELS))
    if unknown:
        raise KeyError(f"Unknown v1.5 model assets: {', '.join(unknown)}")
    models = {
        name: _download(
            **MODELS[name],
            directory=asset_root / name.replace("_", "-"),
        )
        for name in selected
    }
    result = {
        "schema_version": "1.0",
        "release": "v1.5.0",
        "models": models,
        "gated_access_probe": _probe_gated_access(),
        "validation_scope": (
            "Immutable local snapshots for MedUMM adapter acceptance; downloading an "
            "artifact is not runtime validation. Gated releases remain blocked until the "
            "user accepts the upstream terms and provides an authorized token."
        ),
    }
    (asset_root / "provenance.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare pinned MedUMM v1.5 model assets")
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=sorted(MODELS))
    return parser


def main(arguments: list[str] | None = None) -> int:
    values = build_parser().parse_args(arguments)
    print(json.dumps(prepare(values.asset_root, values.models), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
