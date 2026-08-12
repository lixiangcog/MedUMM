from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from medumm.core.io import ensure_directory, write_json


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), child)
        else:
            flattened[prefix] = value

    visit("", metrics)
    return flattened


def build_leaderboard(
    reports: list[str | Path],
    output_directory: str | Path,
) -> dict[str, str]:
    """Aggregate evaluation score files into JSON and CSV leaderboard artifacts."""

    rows = []
    for raw_path in reports:
        path = Path(raw_path)
        report = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "benchmark": report.get("benchmark"),
            "model": report.get("metadata", {}).get("model", "unknown"),
            "dataset_size": report.get("dataset_size"),
            "source": str(path),
            **_flatten_metrics(report.get("metrics", {})),
        })
    directory = ensure_directory(output_directory)
    json_path = write_json(directory / "leaderboard.json", rows)
    csv_path = directory / "leaderboard.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"json_path": str(json_path), "csv_path": str(csv_path)}
