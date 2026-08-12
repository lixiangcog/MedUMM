from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from medumm.core.io import ensure_directory, read_jsonl, write_json, write_jsonl


SHARD_PATTERN = re.compile(r"\.rank-(\d+)-of-(\d+)\.jsonl$")


def merge_prediction_shards(
    shard_paths: list[str | Path],
    output_path: str | Path,
    *,
    expected_count: int | None = None,
) -> dict[str, Any]:
    """Strictly merge rank-local predictions into one deterministic JSONL file."""

    if not shard_paths:
        raise ValueError("At least one prediction shard is required.")
    shards: dict[int, tuple[Path, list[dict[str, Any]]]] = {}
    shard_count: int | None = None
    fingerprints: set[str] = set()
    identifiers: set[str] = set()
    rows: list[dict[str, Any]] = []
    for raw_path in shard_paths:
        path = Path(raw_path)
        match = SHARD_PATTERN.search(path.name)
        if match is None:
            raise ValueError(f"Prediction shard name is invalid: {path.name}.")
        rank, count = (int(value) for value in match.groups())
        if shard_count is not None and count != shard_count:
            raise ValueError("Prediction shards declare different shard counts.")
        shard_count = count
        if rank in shards:
            raise ValueError(f"Duplicate prediction shard rank: {rank}.")
        current = read_jsonl(path)
        shards[rank] = (path, current)
        for row in current:
            sample_id = str(row.get("id", ""))
            if not sample_id:
                raise ValueError(f"Prediction in {path} has no id.")
            if sample_id in identifiers:
                raise ValueError(f"Duplicate prediction id across shards: {sample_id}.")
            identifiers.add(sample_id)
            fingerprints.add(str(row.get("fingerprint", "")))
            rows.append(row)
    expected_ranks = set(range(shard_count or 0))
    if set(shards) != expected_ranks:
        raise ValueError(
            f"Missing prediction shard ranks: {sorted(expected_ranks - set(shards))}."
        )
    if len(fingerprints) != 1 or "" in fingerprints:
        raise ValueError("Prediction shards must share one non-empty fingerprint.")
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} predictions, found {len(rows)}.")
    rows.sort(key=lambda row: str(row["id"]))
    target = Path(output_path)
    ensure_directory(target.parent)
    write_jsonl(target, rows)
    manifest = {
        "schema_version": "1.0",
        "status": "completed",
        "shard_count": shard_count,
        "prediction_count": len(rows),
        "fingerprint": next(iter(fingerprints)),
        "shards": [path.name for _, (path, _) in sorted(shards.items())],
        "output": target.name,
    }
    manifest_path = target.with_name("merge_manifest.json")
    write_json(manifest_path, manifest)
    return {**manifest, "output_path": str(target), "manifest_path": str(manifest_path)}
